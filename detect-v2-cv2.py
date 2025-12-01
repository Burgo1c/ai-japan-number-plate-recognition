"""
License Plate Detection and Recognition System (OpenCV Camera with Edge TPU)

This script uses OpenCV (cv2) to capture video frames from a camera,
then processes them using YOLO object detection optimized for Edge TPU to identify 
license plates and their characters. The detected license plate numbers are 
extracted and displayed in the console.

Author: Liam Burgess
Date: November 19, 2025
Version: 1.3

Dependencies:
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- PyYAML
- PyCoral (for Edge TPU acceleration)
- RPi.GPIO (for LED status indicators)

Hardware Requirements:
- Raspberry Pi with camera module or USB camera
- Coral Edge TPU (for hardware acceleration)
- Dual-color LED (connected to GPIO pins)
- Sufficient processing power for real-time inference

Usage:
- Ensure config.yaml is properly configured
- Run the script: python detect-v2-cv2.py
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
import numpy as np
import time
from ultralytics import YOLO
import threading
import queue
import yaml
import csv
import os
from pathlib import Path
# import RPi.GPIO as GPIO

# # --- LED Configuration ---
# RED_LED_PIN = 27
# GREEN_LED_PIN = 17

# # Set up GPIO pins
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(RED_LED_PIN, GPIO.OUT)
# GPIO.setup(GREEN_LED_PIN, GPIO.OUT)

# # LED control functions
# def led_off():
#     """Turn off both LEDs"""
#     GPIO.output(RED_LED_PIN, GPIO.LOW)
#     GPIO.output(GREEN_LED_PIN, GPIO.LOW)

# def led_green():
#     """Turn on green LED (system working correctly)"""
#     GPIO.output(RED_LED_PIN, GPIO.LOW)
#     GPIO.output(GREEN_LED_PIN, GPIO.HIGH)

# def led_red():
#     """Turn on red LED (system error)"""
#     GPIO.output(RED_LED_PIN, GPIO.HIGH)
#     GPIO.output(GREEN_LED_PIN, GPIO.LOW)

# --- CSV and Video Logging Configuration ---
home_dir = str(Path.home())
csv_file_path = os.path.join(home_dir, "ai-ltms.csv")
video_file_path = os.path.join(home_dir, "ai-ltms-debug.mp4")

def log_to_csv(plate_number, timestamp):
    """
    Log license plate data to CSV file
    
    Args:
        plate_number: Detected license plate number
        timestamp: Time of detection
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

def draw_detections(frame, boxes):
    """
    Draw bounding boxes and class names on the frame
    
    Args:
        frame: The image frame to draw on
        boxes: YOLO detection boxes
        
    Returns:
        Frame with bounding boxes and class names drawn
    """
    # Create a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Draw each detection
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Get class name and confidence
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]
        confidence = float(box.conf[0].item())
        
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
# Uses OpenCV (cv2) for camera handling.
print("Starting camera feed...")
# Initialize LED (off at startup)
# led_off()

# Initialize OpenCV camera
try:
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default camera)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        raise Exception("Could not open video device")
    
    # Give the camera a moment to warm up
    time.sleep(1.0)
    # led_green()  # Show green LED to indicate system is working
except Exception as e:
    print(f"Error: Could not open camera with OpenCV: {e}")
    print("Ensure the camera is connected and accessible.")
    # led_red()  # Show error with red LED
    exit()

# Initialize video writer
fps = 30.0  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (mp4v for MP4 format)
video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

if not video_writer.isOpened():
    print(f"Warning: Could not create video writer. Debug video will not be saved.")
    # led_red()  # Show error with red LED
else:
    print(f"Video recording initialized. Saving to {video_file_path}")

print("Camera started. Running detection loop (press Ctrl+C to stop)...")

threading.Thread(target=worker, daemon=True).start()

last_printed_bottom = ""

try:
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame from camera")
            # led_red()  # Show error with red LED
            break
        
        # Create a copy of the frame for visualization
        display_frame = frame.copy()
        
        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available
        try:
            final_bottom, boxes = result_queue.get_nowait()
            
            # Draw detections on the frame
            if len(boxes) > 0:
                display_frame = draw_detections(display_frame, boxes)
            
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
                
                # Log to CSV file
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
            pass # No new result, just keep looping
        
        # Write the frame to video file
        if video_writer.isOpened():
            video_writer.write(display_frame)

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")
    # led_off()  # Turn off LED on clean exit
except Exception as e:
    # Show red LED on error
    # led_red()
    print(f"Error: {e}")

# --- 6. CLEANUP ---
# Performs necessary cleanup operations when the script is terminated.
# Ensures resources are properly released.
frame_queue.put(None) # Signal worker to exit
cap.release()
if video_writer.isOpened():
    video_writer.release()
    print(f"Video saved to {video_file_path}")
cv2.destroyAllWindows()
# GPIO.cleanup()  # Clean up GPIO pins
# led_off()  # Ensure LEDs are off
print("Script finished.")
