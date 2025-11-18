import cv2
from picamera2 import Picamera2  # <-- ADDED
import numpy as np
import time
from ultralytics import YOLO
import threading
import queue
import yaml

# --- 1. CONFIGURATION ---
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Load Configuration ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        # This 'config' holds your settings
        config = yaml.safe_load(f)
        MODEL_PATH = config['edge_tpu_model_path']
        DATA_YAML_PATH = config['data_yaml_path']
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


# --- 2. INITIALIZE MODELS ---
print(f"Loading YOLO model ({MODEL_PATH})...")
print("This will use the Coral Edge TPU if 'pycoral' is installed.")
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure 'ultralytics' and 'pycoral' are installed")
    print("and the model file is in the correct location.")
    exit()

# --- 3. PARSING FUNCTION ---
def get_yolo_parsed_strings(predictions):
    try:
        plate_detection = next(p for p in predictions if p["class"] == "NumberPLATE")
        divider_y = plate_detection["y"]
    except StopIteration:
        return None

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

    # if len(number_string) == 4:
    #     number_string = f"{number_string[:2]}-{number_string[2:]}"
    
    bottom_string = f"{hiragana_char}{number_string}"
    return bottom_string

# --- 4. WORKER THREAD FOR PROCESSING ---
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # --- YOLO Inference (Offloaded to Coral) ---
        yolo_results = model.predict(source=frame, conf=CONF_THRESHOLD, save=False, verbose=False)

        # Process YOLO results
        yolo_predictions = []
        for box in yolo_results[0].boxes:
            yolo_predictions.append({
                "x": box.xywh[0][0].item(),
                "y": box.xywh[0][1].item(),
                # CORRECT: Uses the class_names list from your YAML
                "class": class_names[int(box.cls[0].item())],
            })

        yolo_bottom = get_yolo_parsed_strings(yolo_predictions)

        # --- Combine and Pass Results ---
        final_bottom = yolo_bottom if yolo_bottom else ""

        # Pass all results to the main thread
        result_queue.put((final_bottom, yolo_results[0].boxes))

# --- 5. MAIN THREAD FOR CAMERA AND CONSOLE ---
print("Starting camera feed...")
# --- REPLACED cv2.VideoCapture ---
picam2 = Picamera2()
try:
    # Create a camera config, using a different name
    # to avoid conflicting with your settings 'config'
    cam_config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "XRGB8888"}
    )
    picam2.configure(cam_config)
    
    picam2.start()
    time.sleep(1.0) # Give camera time to warm up
    print("Camera started. Running detection loop (press Ctrl+C to stop)...")

except Exception as e:
    print(f"Error: Could not open camera with picamera2: {e}")
    print("Ensure the camera is connected and enabled in raspi-config.")
    exit()
# --- END REPLACEMENT ---

threading.Thread(target=worker, daemon=True).start()

last_printed_bottom = ""

try:
    while True:
        # --- REPLACED cap.read() ---
        # picamera2 gives an RGB array
        frame_rgb = picam2.capture_array()
        # OpenCV/YOLO expects BGR, so we must convert
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # --- END REPLACEMENT ---

        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available
        try:
            final_bottom, boxes = result_queue.get_nowait()
            
            # --- CONSOLE OUTPUT ---
            new_result = False
            if final_bottom and final_bottom != last_printed_bottom:
                last_printed_bottom = final_bottom
                new_result = True

            if new_result:
                print("--- Plate Detected ---")
                print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Bottom: {last_printed_bottom}")
                print("----------------------")
            
        except queue.Empty:
            pass # No new result, just keep looping

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")

# --- 6. CLEANUP ---
frame_queue.put(None) # Signal worker to exit
# --- REPLACED cap.release() ---
picam2.stop()
# --- END REPLACEMENT ---
print("Script finished.")