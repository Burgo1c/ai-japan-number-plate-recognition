"""
License Plate Detection and Recognition System (Raspberry Pi Camera Version)

This script uses a Raspberry Pi camera with the Picamera2 library to capture video frames,
then processes them using YOLO object detection to identify license plates and their characters.
The detected license plate numbers are extracted and displayed in the console.

Author: Liam Burgess
Date: November 19, 2025
Version: 1.0

Dependencies:
- OpenCV (cv2)
- Picamera2
- NumPy
- Ultralytics YOLO
- PyYAML

Hardware Requirements:
- Raspberry Pi with camera module
- Sufficient processing power for real-time inference

Usage:
- Ensure config.yaml is properly configured
- Run the script: python detect.py
- Press Ctrl+C to exit
"""

import cv2
from picamera2 import Picamera2
import numpy as np
import time
from ultralytics import YOLO
import threading
import queue
import yaml

# --- 1. CONFIGURATION ---
# This section handles loading configuration settings and initializing constants
# for the license plate detection system.
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Load Configuration ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        # This 'config' variable holds all your settings
        config = yaml.safe_load(f)
        MODEL_PATH = config['model_path']
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
# Loads the YOLO model specified in the configuration file.
# This model is responsible for detecting license plates and characters.
print(f"Loading YOLO model ({MODEL_PATH})...")
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure 'ultralytics' is installed (pip install ultralytics)")
    print("and the model file is in the correct location.")
    exit()

# --- 3. PARSING FUNCTION ---
# This function processes YOLO detection results to extract license plate text.
# It separates characters based on their vertical position (above/below divider)
# and combines them according to configuration settings.
def get_yolo_parsed_strings(predictions):
    """
    Extracts and formats license plate text from YOLO detection predictions.
    
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
            # Accessing the main 'config' dictionary, which is now safe
            if config['use_hiragana']:
                hiragana_char = config['location_dictionary'].get(p["class"], p["class"])
            else:
                hiragana_char = ""

    # if len(number_string) == 4:
    #     number_string = f"{number_string[:2]}-{number_string[2:]}"
    
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

# --- 4. WORKER THREAD FOR PROCESSING ---
# Sets up a separate thread for processing frames to maintain real-time performance.
# This allows camera capture to continue while inference is performed in parallel.
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    """
    Worker thread function that processes frames from the queue.
    Performs YOLO inference and extracts license plate text.
    Results are placed in the result_queue for the main thread to consume.
    """
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # --- YOLO Inference ---
        yolo_results = model.predict(source=frame, conf=CONF_THRESHOLD, save=False, verbose=False)

        # Process YOLO results
        yolo_predictions = []
        for box in yolo_results[0].boxes:
            yolo_predictions.append({
                "x": box.xywh[0][0].item(),
                "y": box.xywh[0][1].item(),
                # FIXED: Use 'class_names' loaded from YAML, not 'model.names'
                "class": class_names[int(box.cls[0].item())],
            })

        yolo_bottom = get_yolo_parsed_strings(yolo_predictions)

        # --- Combine and Pass Results ---
        final_bottom = yolo_bottom if yolo_bottom else ""

        # Pass all results to the main thread
        result_queue.put((final_bottom, yolo_results[0].boxes))

# --- 5. MAIN THREAD FOR CAMERA AND CONSOLE ---
# Handles camera initialization, frame capture, and result display.
# Uses Picamera2 library specifically designed for Raspberry Pi cameras.
print("Starting camera feed...")
picam2 = Picamera2()
try:
    # FIXED: Renamed 'config' to 'cam_config' to avoid conflict
    cam_config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "XRGB8888"}
    )
    picam2.configure(cam_config)
    
    # Start the camera
    picam2.start()
    
    # Give the camera a moment to warm up and set auto-exposure
    time.sleep(1.0) 
    print("Camera started. Running detection loop (press Ctrl+C to stop)...")

except Exception as e:
    print(f"Error: Could not open camera with picamera2: {e}")
    print("Ensure the camera is connected and enabled in raspi-config.")
    exit()

threading.Thread(target=worker, daemon=True).start()

# Keep track of the last printed result to avoid spam
last_printed_bottom = ""
# REMOVED: Unused 'last_printed_top'

try:
    while True:
        # picamera2 gives an RGB array, OpenCV (and your model) expects BGR
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available and draw
        try:
            plate_string, boxes = result_queue.get_nowait()
            
            # --- CONSOLE OUTPUT ---
            new_result = False
            if plate_string and plate_string != last_printed_bottom:
                last_printed_bottom = plate_string
                new_result = True

            if new_result:
                print("--- Plate Detected ---")
                print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Plate:  {last_printed_bottom}")
                print("----------------------")
            
        except queue.Empty:
            pass # No new result, just keep looping

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")

# --- 6. CLEANUP ---
# Performs necessary cleanup operations when the script is terminated.
# Ensures resources are properly released.
frame_queue.put(None) # Signal worker to exit
picam2.stop()
print("Script finished.")
