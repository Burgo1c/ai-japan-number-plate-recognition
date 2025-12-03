"""
License Plate Detection and Recognition System (Raspberry Pi Camera Version with ONNX)

This script uses a Raspberry Pi camera with the Picamera2 library to capture video frames,
then processes them using ONNX Runtime to identify license plates and their characters. 
The detected license plate numbers are extracted and displayed in the console.

Author: Liam Burgess
Date: December 3, 2025
Version: 1.0 (ONNX)

Dependencies:
- OpenCV (cv2)
- Picamera2
- NumPy
- ONNX Runtime
- PyYAML
- Pillow (PIL)

Hardware Requirements:
- Raspberry Pi with camera module

Usage:
- Ensure config.yaml is properly configured
- Run the script: python3 detect-onnx.py
- Press Ctrl+C to exit
"""

import cv2
from picamera2 import Picamera2
import numpy as np
import time
import threading
import queue
import yaml
from PIL import Image
import onnxruntime as ort
import os
from datetime import datetime
import csv

# --- 1. CONFIGURATION ---
# This section handles loading configuration settings and initializing constants
# for the license plate detection system.
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_INPUT_SIZE = 640  # ONNX model input size (640x640)

# --- Load Configuration ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        MODEL_PATH = config['onnx_model_path']
        DATA_YAML_PATH = config['data_yaml_path']
        SHUTTER_SPEED_US = config.get('shutter_speed_us', 1000)  # Default 1/1000s
        ANALOGUE_GAIN = config.get('analogue_gain', 2.0)  # Default gain
except FileNotFoundError:
    print("Error: 'config.yaml' not found.")
    print("Please ensure the config.yaml file is in the correct location.")
    exit()
except Exception as e:
    print(f"Error loading config.yaml: {e}")
    exit()

# --- Load Class Names ---
try:
    with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names']
except FileNotFoundError:
    print(f"Error: '{DATA_YAML_PATH}' not found.")
    print("Please ensure the data.yaml file is in the correct location.")
    exit()
except Exception as e:
    print(f"Error loading {DATA_YAML_PATH}: {e}")
    exit()

# --- Create output directories ---
if config.get('enable_recording', False):
    output_dir = config.get('output_directory', 'recordings')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

if config.get('enable_csv_logging', False):
    csv_dir = config.get('csv_output_directory', 'logs')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        print(f"Created CSV log directory: {csv_dir}")


# --- 2. INITIALIZE ONNX MODEL ---
# Loads the ONNX model using ONNX Runtime
print(f"Loading ONNX model ({MODEL_PATH})...")
try:
    # Initialize ONNX Runtime session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create inference session (CPU execution provider)
    ort_session = ort.InferenceSession(
        MODEL_PATH,
        session_options,
        providers=['CPUExecutionProvider']
    )
    
    # Get input and output details
    input_details = ort_session.get_inputs()
    output_details = ort_session.get_outputs()
    
    print("ONNX model loaded successfully.")
    print(f"Input name: {input_details[0].name}")
    print(f"Input shape: {input_details[0].shape}")
    print(f"Input type: {input_details[0].type}")
    print(f"Number of outputs: {len(output_details)}")
    for i, output in enumerate(output_details):
        print(f"Output {i}: name={output.name}, shape={output.shape}, type={output.type}")
    
except Exception as e:
    print(f"Error loading ONNX model from {MODEL_PATH}: {e}")
    print("Please ensure:")
    print("  1. The ONNX model file exists at the specified path")
    print("  2. onnxruntime is installed (pip install onnxruntime)")
    exit()


# --- 3. PREPROCESSING FUNCTION ---
def preprocess_image(image):
    """
    Preprocesses an image for ONNX inference.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Preprocessed image ready for inference
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image_rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    
    # Convert to FLOAT32 and normalize to [0, 1]
    # ONNX models typically expect float input normalized to 0-1 range
    input_data = image_resized.astype(np.float32) / 255.0
    
    # Transpose from HWC to CHW format (channels first)
    input_data = np.transpose(input_data, (2, 0, 1))
    
    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data


# --- 4. INFERENCE FUNCTION ---
def run_inference(image):
    """
    Runs inference on the ONNX model and applies NMS.
    """
    # Get input name
    input_name = input_details[0].name
    
    # Run inference
    outputs = ort_session.run(None, {input_name: image})
    
    # Get output tensor (first output)
    output_data = outputs[0][0]  # Remove batch dimension
    
    # --- Handle Transposed Output ---
    # YOLOv8/v5 ONNX exports are often [Classes+4, Boxes] (e.g., [84, 8400]).
    # We need them as [Boxes, Classes+4] (e.g., [8400, 84]).
    if output_data.shape[0] < output_data.shape[1]:
        output_data = output_data.transpose()

    # Lists to hold candidate detections for NMS
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over all rows (candidates)
    for detection in output_data:
        # Ignore rows that are too short (sanity check)
        if len(detection) < 5:
            continue

        # Extract box and scores
        # columns: 0=x, 1=y, 2=w, 3=h, 4...=class_scores
        box = detection[:4]
        class_scores = detection[4:]
        
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        # Apply sigmoid if your model exports raw logits (common in YOLOv8)
        # If your model already outputs 0-1 probabilities, you can remove this.
        confidence = 1 / (1 + np.exp(-confidence))

        if confidence > CONF_THRESHOLD:
            # Scale box to pixel coordinates
            cx, cy, w, h = box
            
            # Convert center-x/y to top-left-x/y for NMS
            x = int((cx - w / 2) * MODEL_INPUT_SIZE)
            y = int((cy - h / 2) * MODEL_INPUT_SIZE)
            w_pixel = int(w * MODEL_INPUT_SIZE)
            h_pixel = int(h * MODEL_INPUT_SIZE)
            
            boxes.append([x, y, w_pixel, h_pixel])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

    # --- Apply Non-Maximum Suppression (NMS) ---
    # This removes overlapping boxes, keeping only the best one per object.
    nms_threshold = 0.4  
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, nms_threshold)

    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            # Check if class_id is valid for your yaml
            cid = class_ids[i]
            if cid < len(class_names):
                final_detections.append({
                    "x": boxes[i][0] + (boxes[i][2] / 2), # convert back to center X
                    "y": boxes[i][1] + (boxes[i][3] / 2), # convert back to center Y
                    "class": class_names[cid],
                    "score": confidences[i]
                })

    return final_detections

# --- 5. PARSING FUNCTION ---
# This function processes detection results to extract license plate text.
# It separates characters based on their vertical position (above/below divider)
# and combines them according to configuration settings.
def get_yolo_parsed_strings(predictions):
    """
    Extracts and formats license plate text from detection predictions.
    
    Args:
        predictions: List of dictionaries containing detection results with class, x, y coordinates
        
    Returns:
        Formatted string of the license plate text or None if no plate detected
    """
    try:
        plate_detection = next(p for p in predictions if p["class"] == "NumberPLATE")
        divider_y = plate_detection["y"]
    except StopIteration:
        return None

    # Process bottom row (characters below or at the divider)
    bottom_row_detections = sorted([p for p in predictions if p["class"] != "NumberPLATE" and p["y"] >= divider_y], key=lambda p: p["x"])

    hiragana_char = ""
    number_string = ""
    for p in bottom_row_detections:
        if p["class"].isdigit():
            number_string += p["class"]
        else:
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
        
        # Return combined string if top row exists
        if top_string:
            return f"{top_string}{bottom_string}"
    
    # Return only bottom row if get_top_line is false or no top characters detected
    return bottom_string


# --- 6. WORKER THREAD FOR PROCESSING ---
# Sets up a separate thread for processing frames to maintain real-time performance.
# This allows camera capture to continue while inference is performed in parallel.
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    """
    Worker thread function that processes frames from the queue.
    Performs ONNX inference and extracts license plate text.
    Results are placed in the result_queue for the main thread to consume.
    """
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Preprocess image for ONNX
        input_data = preprocess_image(frame)
        
        # Run inference on ONNX model
        detections = run_inference(input_data)
        
        # Parse detections to get license plate text
        plate_text = get_yolo_parsed_strings(detections)
        
        # Pass results to the main thread
        final_bottom = plate_text if plate_text else ""
        result_queue.put((final_bottom, detections))


# --- 7. MAIN THREAD FOR CAMERA AND CONSOLE ---
# Handles camera initialization, frame capture, and result display.
# Uses Picamera2 library specifically designed for Raspberry Pi cameras.
print("Starting camera feed...")
picam2 = Picamera2()
try:
    cam_config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "XRGB8888"}
    )
    picam2.configure(cam_config)
    
    # Set high shutter speed and gain before starting camera
    print(f"Setting shutter speed to {SHUTTER_SPEED_US} microseconds (1/{1000000//SHUTTER_SPEED_US}s)")
    print(f"Setting analogue gain to {ANALOGUE_GAIN}")
    
    picam2.set_controls({
        "ExposureTime": SHUTTER_SPEED_US,  # Fixed shutter speed in microseconds
        "AnalogueGain": ANALOGUE_GAIN,     # Gain to compensate for fast shutter
        "AeEnable": False                   # Disable auto-exposure
    })
    
    # Start the camera
    picam2.start()
    
    # Give the camera a moment to stabilize
    time.sleep(1.0) 
    print("Camera started. Running detection loop (press Ctrl+C to stop)...")

except Exception as e:
    print(f"Error: Could not open camera with picamera2: {e}")
    print("Ensure the camera is connected and enabled in raspi-config.")
    exit()

threading.Thread(target=worker, daemon=True).start()

# --- Initialize Video Writer ---
video_writer = None
if config.get('enable_recording', False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.get('output_directory', 'recordings')
    video_filename = os.path.join(output_dir, f"recording_{timestamp}.mp4")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*config.get('video_codec', 'mp4v'))
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
    
    if video_writer.isOpened():
        print(f"Recording to: {video_filename}")
    else:
        print(f"Warning: Could not initialize video writer for {video_filename}")
        video_writer = None

# --- Initialize CSV Logger ---
csv_file = None
csv_writer = None
if config.get('enable_csv_logging', False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = config.get('csv_output_directory', 'logs')
    csv_filename = os.path.join(csv_dir, f"detections_{timestamp}.csv")
    
    try:
        csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Plate Number'])
        print(f"Logging detections to: {csv_filename}")
    except Exception as e:
        print(f"Warning: Could not initialize CSV logger: {e}")
        csv_file = None
        csv_writer = None

# Function to draw bounding boxes on frame
def draw_bounding_boxes(frame, detections):
    """
    Draws bounding boxes on the frame for detected objects.
    
    Args:
        frame: The frame to draw on
        detections: List of detection dictionaries
        
    Returns:
        Frame with bounding boxes drawn
    """
    if not config.get('draw_bounding_boxes', True):
        return frame
    
    frame_with_boxes = frame.copy()
    thickness = config.get('bbox_thickness', 2)
    plate_color = tuple(config.get('plate_bbox_color', [0, 255, 0]))
    char_color = tuple(config.get('character_bbox_color', [255, 0, 0]))
    
    # Scale factor from model input size to frame size
    scale_x = FRAME_WIDTH / MODEL_INPUT_SIZE
    scale_y = FRAME_HEIGHT / MODEL_INPUT_SIZE
    
    for detection in detections:
        # Get detection info
        class_name = detection['class']
        confidence = detection['score']
        
        # Scale coordinates to frame size
        cx = int(detection['x'] * scale_x)
        cy = int(detection['y'] * scale_y)
        
        # Estimate box size (approximate, since we don't have exact w/h in final_detections)
        # Using a fixed size based on typical character/plate dimensions
        if class_name == "NumberPLATE":
            w = int(100 * scale_x)
            h = int(40 * scale_y)
        else:
            w = int(20 * scale_x)
            h = int(30 * scale_y)
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        # Choose color based on class
        color = plate_color if class_name == "NumberPLATE" else char_color
        
        # Draw rectangle
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 10, label_size[1])
        
        # Draw background for text
        cv2.rectangle(frame_with_boxes, 
                     (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0], label_y + 5), 
                     color, -1)
        
        # Draw text
        cv2.putText(frame_with_boxes, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_with_boxes

# Keep track of the last printed result to avoid spam
last_printed_bottom = ""
# Keep track of the latest frame with boxes for recording
latest_frame_with_boxes = None

try:
    while True:
        # picamera2 gives an RGB array, OpenCV expects BGR
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available
        try:
            plate_string, detections = result_queue.get_nowait()
            
            # Draw bounding boxes on the frame
            latest_frame_with_boxes = draw_bounding_boxes(frame, detections)
            
            # --- CONSOLE OUTPUT ---
            new_result = False
            if plate_string and plate_string != last_printed_bottom:
                last_printed_bottom = plate_string
                new_result = True

            if new_result:
                current_time = datetime.now()
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                
                print("--- Plate Detected ---")
                print(f"Time:   {timestamp_str}")
                print(f"Plate:  {last_printed_bottom}")
                print("----------------------")
                
                # Log to CSV if enabled
                if csv_writer is not None:
                    csv_writer.writerow([timestamp_str, last_printed_bottom])
                    csv_file.flush()  # Ensure data is written immediately
            
        except queue.Empty:
            # No new result, use the last frame with boxes if available
            if latest_frame_with_boxes is None:
                latest_frame_with_boxes = frame
        
        # Write frame to video if recording is enabled
        if video_writer is not None and latest_frame_with_boxes is not None:
            video_writer.write(latest_frame_with_boxes)

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")

# --- 8. CLEANUP ---
# Performs necessary cleanup operations when the script is terminated.
# Ensures resources are properly released.
frame_queue.put(None)  # Signal worker to exit
picam2.stop()

# Release video writer if it was initialized
if video_writer is not None:
    video_writer.release()
    print("Video recording saved successfully.")

# Close CSV file if it was initialized
if csv_file is not None:
    csv_file.close()
    print("CSV log file saved successfully.")

print("Script finished.")
