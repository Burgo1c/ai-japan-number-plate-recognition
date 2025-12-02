"""
License Plate Detection and Recognition System (Edge TPU Optimized Version)

This script uses a Raspberry Pi camera with the Picamera2 library to capture video frames,
then processes them using YOLO object detection optimized for Edge TPU to identify 
license plates and their characters. The detected license plate numbers are 
extracted and displayed in the console.

Author: Liam Burgess
Date: November 19, 2025
Version: 1.3

Dependencies:
- OpenCV (cv2)
- Picamera2
- NumPy
- PyYAML
- ai_edge_litert (for Edge TPU acceleration)
- tensorflow (for Edge TPU acceleration)
- RPi.GPIO (for LED status indicators)

Hardware Requirements:
- Raspberry Pi with camera module
- Coral Edge TPU (for hardware acceleration)
- Dual-color LED (connected to GPIO pins)
- Sufficient processing power for real-time inference

Usage:
- Ensure config.yaml is properly configured
- Run the script: python detect-v2.py
- Press Ctrl+C to exit

LED Status Indicators:
- Green: System working correctly
- Red: System error/crash
- Off: System not active

CSV Logging:
- License plate numbers and timestamps are logged to ~/ai-ltms.csv

Video Recording:
- Processed video with bounding boxes and class names is saved to ~/ai-ltms-debug.mp4
"""
import cv2
from picamera2 import Picamera2
import numpy as np
import time
import tflite_runtime.interpreter as tflite
import threading
import queue
import yaml
import csv
import os
from pathlib import Path

# --- Configuration (GPIO commented out for platform compatibility) ---
# ... (LED GPIO configuration remains commented out)
# ... (LED control functions remain commented out)

# --- CSV and Video Logging Configuration ---
home_dir = str(Path.home())
csv_file_path = os.path.join(home_dir, "ai-ltms.csv")
video_file_path = os.path.join(home_dir, "ai-ltms-debug.mp4")

def log_to_csv(plate_number, timestamp):
    """
    Log license plate data to CSV file
    """
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'plate_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'timestamp': timestamp,
            'plate_number': plate_number
        })

def draw_detections(frame, boxes_for_drawing, class_names):
    """
    Draw bounding boxes and class names on the frame using the new list of dicts format.
    
    Args:
        frame: The image frame to draw on
        boxes_for_drawing: List of dictionaries: 
                           [{"box": [x1, y1, x2, y2], "class_id": 0, "confidence": 0.95}]
        class_names: List of all class names
        
    Returns:
        Frame with bounding boxes and class names drawn
    """
    # Create a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Draw each detection
    for detection in boxes_for_drawing:
        # Get box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = detection['box']
        
        # Get class name and confidence
        class_id = detection['class_id']
        class_name = class_names[class_id]
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated_frame

# --- 1. CONFIGURATION ---
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45 # Added IOU threshold for NMS
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Load Configuration ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        MODEL_PATH = config['edge_tpu_model_path']
        DATA_YAML_PATH = config['data_yaml_path']
except FileNotFoundError:
    print("Error: 'config.yaml' not found.")
    exit()
except Exception as e:
    print(f"Error loading config.yaml: {e}")
    exit()

# --- Load Class Names ---
try:
    with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names']
        NUM_CLASSES = len(class_names)
except FileNotFoundError:
    print(f"Error: '{DATA_YAML_PATH}' not found.")
    exit()
except Exception as e:
    print(f"Error loading {DATA_YAML_PATH}: {e}")
    exit()


# --- 2. INITIALIZE MODELS (Using ai-edge-litert) ---
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

print(f"Loading TFLite model ({MODEL_PATH}) with Edge TPU delegate...")
try:
    # Load the Delegate specifically for tflite-runtime
    # The 'libedgetpu.so.1' library is installed by libedgetpu1-std
    delegate = tflite.load_delegate('libedgetpu.so.1')
    
    # Create the interpreter
    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[delegate]
    )
    interpreter.allocate_tensors()
    
    # Get I/O details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    INPUT_SHAPE = input_details[0]['shape'][1:3] # H, W
    
    print("Edge TPU Delegate loaded successfully.")

except ValueError as e:
    print("Error: Could not load Edge TPU delegate.")
    print("Ensure you installed: sudo apt install libedgetpu1-std")
    print(f"Details: {e}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- POST-PROCESSING FUNCTIONS (REQUIRED) ---

def xywh2xyxy(x):
    """
    Convert YOLO box format (x_center, y_center, width, height) to 
    (x_min, y_min, x_max, y_max)
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def post_process_tflite(raw_outputs, input_shape, frame_shape, class_names, conf_threshold, iou_threshold):
    """
    Converts raw YOLOv8 TFLite model output tensors into usable bounding boxes.
    Applies confidence filtering, coordinate scaling, and Non-Maximum Suppression (NMS).
    
    Args:
        raw_outputs (list): List containing the single output array from the interpreter.
        input_shape (tuple): (H, W) of the model's input (e.g., 640, 640).
        frame_shape (tuple): (W, H) of the original camera frame (e.g., 640, 480).
        class_names (list): List of all class names.
        conf_threshold (float): Confidence threshold for filtering.
        iou_threshold (float): IOU threshold for NMS.
        
    Returns:
        tuple: (yolo_predictions, boxes_for_drawing)
    """
    
    # YOLOv8 TFLite output is typically (1, 84, N_ANCHORS) -> transposed to (1, N_ANCHORS, 84)
    # The output format is: [x_center, y_center, w, h, class_0_conf, class_1_conf, ...]
    
    output = raw_outputs[0].squeeze().T # Shape: (N_ANCHORS, 4 + NUM_CLASSES)
    
    # 1. Separate Bounding Box and Confidence/Class Data
    # Assuming the first 4 values are box coordinates (xc, yc, w, h)
    boxes = output[:, :4]
    scores = output[:, 4:]
    
    # 2. Find best class and confidence score
    max_scores = np.amax(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # 3. Apply Confidence Threshold
    mask = max_scores >= conf_threshold
    boxes_filtered = boxes[mask]
    scores_filtered = max_scores[mask]
    class_ids_filtered = class_ids[mask]
    
    # 4. Convert boxes from (xc, yc, w, h) to (x_min, y_min, x_max, y_max)
    boxes_xyxy = xywh2xyxy(boxes_filtered)
    
    # 5. Coordinate Scaling (from model size to frame size)
    # This scaling assumes no letterboxing was used, only direct resizing.
    model_h, model_w = input_shape
    frame_w, frame_h = frame_shape
    
    x_scale = frame_w / model_w
    y_scale = frame_h / model_h
    
    boxes_scaled = np.zeros_like(boxes_xyxy, dtype=np.float32)
    boxes_scaled[:, 0] = boxes_xyxy[:, 0] * x_scale # x_min
    boxes_scaled[:, 1] = boxes_xyxy[:, 1] * y_scale # y_min
    boxes_scaled[:, 2] = boxes_xyxy[:, 2] * x_scale # x_max
    boxes_scaled[:, 3] = boxes_xyxy[:, 3] * y_scale # y_max
    
    # Convert to integer coordinates for OpenCV/NMS
    boxes_scaled = boxes_scaled.astype(np.int32)
    
    # 6. Non-Maximum Suppression (NMS)
    # OpenCV's NMSBoxes expects (x, y, w, h) format for boxes
    boxes_for_nms = np.zeros_like(boxes_scaled, dtype=np.int32)
    boxes_for_nms[:, 0] = boxes_scaled[:, 0] # x
    boxes_for_nms[:, 1] = boxes_scaled[:, 1] # y
    boxes_for_nms[:, 2] = boxes_scaled[:, 2] - boxes_scaled[:, 0] # w
    boxes_for_nms[:, 3] = boxes_scaled[:, 3] - boxes_scaled[:, 1] # h
    
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), 
        scores_filtered.tolist(), 
        conf_threshold, 
        iou_threshold
    )
    
    # NMSBoxes returns indices as a flat list if successful
    if len(indices) == 0:
        return [], []
        
    final_indices = indices.flatten()

    # 7. Final Formatting
    final_boxes_xyxy = boxes_scaled[final_indices]
    final_scores = scores_filtered[final_indices]
    final_classes = class_ids_filtered[final_indices]

    yolo_predictions = []
    boxes_for_drawing = []

    for box_xyxy, score, class_id in zip(final_boxes_xyxy, final_scores, final_classes):
        
        class_name = class_names[class_id]
        x_center = int((box_xyxy[0] + box_xyxy[2]) / 2) # needed for plate parsing
        y_center = int((box_xyxy[1] + box_xyxy[3]) / 2) # needed for plate parsing

        # Structure for plate text parsing (get_yolo_parsed_strings)
        yolo_predictions.append({
            "class": class_name,
            "x": x_center,
            "y": y_center,
            "confidence": float(score)
        })
        
        # Structure for drawing (draw_detections)
        boxes_for_drawing.append({
            "box": box_xyxy.tolist(),
            "class_id": int(class_id),
            "confidence": float(score)
        })

    return yolo_predictions, boxes_for_drawing


# --- 3. PARSING FUNCTION ---
# This function processes YOLO detection results to extract license plate text.
# NOTE: This function requires the "yolo_predictions" input structure defined in post_process_tflite.
def get_yolo_parsed_strings(predictions):
    """
    Extracts and formats license plate text from YOLO detection predictions.
    
    Args:
        predictions: List of dictionaries containing detection results with class, x, y coordinates
        
    Returns:
        Formatted string of the license plate text or None if no plate detected
    """
    # ... (Your existing parsing logic is correct for the new predictions format)
    try:
        # Find the main license plate box to establish the divider Y-coordinate
        plate_detection = next(p for p in predictions if p["class"] == "NumberPLATE")
        divider_y = plate_detection["y"]
    except StopIteration:
        return None

    # Sort characters below the divider by their X-coordinate
    bottom_row_detections = sorted([p for p in predictions if p["class"] != "NumberPLATE" and p["y"] >= divider_y], key=lambda p: p["x"])

    hiragana_char = ""
    number_string = ""
    for p in bottom_row_detections:
        if p["class"].isdigit():
            number_string += p["class"]
        else:
            # Safely accesses the main 'config' variable
            if config['use_hiragana']:
                hiragana_char = config['location_dictionary'].get(p["class"], p["class"])
            else:
                hiragana_char = ""

    bottom_string = f"{hiragana_char}{number_string}"
        
    # Process top row if configured
    if config['get_top_line']:
        # Get characters above the divider
        top_row_detections = sorted([p for p in predictions if p["class"] != "NumberPLATE" and p["y"] < divider_y], key=lambda p: p["x"])
        
        top_string = ""
        for p in top_row_detections:
            if p["class"].isdigit():
                top_string += p["class"]
            else:
                top_string += config['location_dictionary'].get(p["class"], p["class"])
        
        if top_string:
            return f"{top_string}{bottom_string}"
    
    return bottom_string

# --- 4. WORKER THREAD FOR PROCESSING ---
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    """
    Worker thread function that processes frames from the queue using TFLite-Runtime.
    """
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # 1. Pre-process the frame (Resize and add batch dim)
        input_tensor = cv2.resize(frame, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
        input_tensor = input_tensor[np.newaxis, ...]
        
        # Check for model input type (Assuming uint8 for Edge TPU)
        if input_details[0]['dtype'] == np.uint8:
            input_tensor = input_tensor.astype(np.uint8) 
        
        # 2. Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)

        # 3. Invoke Inference (Offloaded to Coral)
        interpreter.invoke()

        # 4. Get the raw output tensors
        raw_outputs = []
        for output in output_details:
            # NOTE: For quantized models, the output tensor may need de-quantization
            # by checking output_details[i]['quantization']. 
            # This implementation assumes the Edge TPU compiler handles it or output is float.
            raw_outputs.append(interpreter.get_tensor(output['index']))
        
        # 5. Post-process (Decoding, NMS, Scaling)
        yolo_predictions, boxes_for_drawing = post_process_tflite(
            raw_outputs, 
            INPUT_SHAPE, 
            (FRAME_WIDTH, FRAME_HEIGHT), 
            class_names, 
            CONF_THRESHOLD,
            IOU_THRESHOLD # Pass the new IOU threshold
        )
        
        # 6. Parse and format final output text
        yolo_bottom = get_yolo_parsed_strings(yolo_predictions)

        # --- Combine and Pass Results ---
        final_bottom = yolo_bottom if yolo_bottom else ""

        # Pass all results to the main thread
        result_queue.put((final_bottom, boxes_for_drawing))

# --- 5. MAIN THREAD FOR CAMERA AND CONSOLE ---
print("Starting camera feed...")
# led_off() # Initialize LED (off at startup)

# Initialize Picamera2
picam2 = Picamera2()
try:
    cam_config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(cam_config)
    
    picam2.start()
    time.sleep(1.0)
    # led_green()
except Exception as e:
    print(f"Error: Could not open camera with picamera2: {e}")
    # led_red()
    exit()

# Initialize video writer
fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

if not video_writer.isOpened():
    print(f"Warning: Could not create video writer. Debug video will not be saved.")
else:
    print(f"Video recording initialized. Saving to {video_file_path}")

print("Camera started. Running detection loop (press Ctrl+C to stop)...")

threading.Thread(target=worker, daemon=True).start()

last_printed_bottom = ""

try:
    while True:
        # Picamera2 gives an RGB array, OpenCV (and your model) expects BGR for visualization
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        display_frame = frame.copy()
        
        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available
        try:
            final_bottom, boxes = result_queue.get_nowait()
            
            # Draw detections on the frame (uses new signature)
            if len(boxes) > 0:
                display_frame = draw_detections(display_frame, boxes, class_names)
            
            # --- CONSOLE OUTPUT ---
            new_result = False
            if final_bottom and final_bottom != last_printed_bottom:
                last_printed_bottom = final_bottom
                new_result = True

            if new_result:
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                print("--- Plate Detected ---")
                print(f"Time:   {current_time}")
                print(f"Bottom: {last_printed_bottom}")
                print("----------------------")
                
                log_to_csv(last_printed_bottom, current_time)
            
            # Add timestamp to the frame
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(display_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detected plate number to the frame if available
            if final_bottom:
                cv2.putText(display_frame, f"Plate: {final_bottom}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except queue.Empty:
            # Add timestamp to the frame even when no detections
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(display_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            pass 
        
        # Write the frame to video file
        if video_writer.isOpened():
            video_writer.write(display_frame)

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")
    # led_off()
except Exception as e:
    # led_red()
    print(f"Error: {e}")

# --- 6. CLEANUP ---
frame_queue.put(None)
picam2.stop()
if video_writer.isOpened():
    video_writer.release()
    print(f"Video saved to {video_file_path}")
cv2.destroyAllWindows()
# GPIO.cleanup()
print("Script finished.")
