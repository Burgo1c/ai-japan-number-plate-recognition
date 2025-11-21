"""
License Plate Detection and Recognition System (Edge TPU Optimized Version)

This script uses a standard USB camera with OpenCV to capture video frames,
then processes them using YOLO object detection optimized for Edge TPU to identify 
license plates and their characters. The detected license plate numbers are 
extracted and displayed in the console.

Author: Liam Burgess
Date: November 19, 2025
Version: 1.0

Dependencies:
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- PyYAML
- PyCoral (for Edge TPU acceleration)

Hardware Requirements:
- USB camera
- Coral Edge TPU (for hardware acceleration)
- Sufficient processing power for real-time inference

Usage:
- Ensure config.yaml is properly configured
- Run the script: python detect-v2.py
- Press Ctrl+C to exit
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
import queue
import yaml

# --- 1. CONFIGURATION ---
# This section handles loading configuration settings and initializing constants
# for the license plate detection system. This version uses Edge TPU optimized models.
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
# Loads the YOLO model optimized for Edge TPU specified in the configuration file.
# This model is responsible for detecting license plates and characters with hardware acceleration.
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
        return divider_y
    except StopIteration:
        return None

# --- 4. WORKER THREAD FOR PROCESSING ---
# Sets up a separate thread for processing frames to maintain real-time performance.
# This allows camera capture to continue while inference is performed in parallel.
# The Edge TPU accelerates the inference process for better performance.
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    """
    Worker thread function that processes frames from the queue.
    Performs YOLO inference using Edge TPU acceleration and extracts license plate text.
    Results are placed in the result_queue for the main thread to consume.
    """
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
# Handles camera initialization, frame capture, and result display.
# Uses OpenCV's VideoCapture for standard USB camera support.
print("Starting camera feed...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Camera started. Running detection loop (press Ctrl+C to stop)...")

threading.Thread(target=worker, daemon=True).start()

last_printed_bottom = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available
        try:
            final_bottom, boxes = result_queue.get_nowait()
            
            # --- CONSOLE OUTPUT ---
            if final_bottom:
                print("--- Plate Detected ---")
                print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"divider_y: {final_bottom}")
                print("----------------------")
            
        except queue.Empty:
            pass # No new result, just keep looping

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")

# --- 6. CLEANUP ---
# Performs necessary cleanup operations when the script is terminated.
# Ensures resources are properly released.
frame_queue.put(None) # Signal worker to exit
cap.release()
cv2.destroyAllWindows()
print("Script finished.")
