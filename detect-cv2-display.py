"""
License Plate Detection and Recognition System with Video Display

This script uses a standard USB camera with OpenCV to capture video frames,
then processes them using YOLO object detection to identify license plates and their characters.
The detected license plate numbers are extracted and displayed in both the console and a video window.

Author: Liam Burgess
Date: November 19, 2025
Version: 1.1 (with video display)

Dependencies:
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- PyYAML

Hardware Requirements:
- USB camera
- Sufficient processing power for real-time inference

Usage:
- Ensure config.yaml is properly configured
- Run the script: python detect-cv2-display.py
- Press 'q' or ESC to exit
- Press 'f' to toggle fullscreen mode
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
import queue
import yaml
import os
from datetime import datetime

# --- 1. CONFIGURATION ---
# This section handles loading configuration settings and initializing constants
# for the license plate detection system. Uses standard YOLO model with USB camera.
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

# --- Create output directory for recordings ---
if config.get('enable_recording', False):
    output_dir = config.get('output_directory', 'recordings')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


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

# --- 5. MAIN THREAD FOR CAMERA AND DISPLAY ---
# Handles camera initialization, frame capture, result display, and video window.
# Uses OpenCV's VideoCapture for standard USB camera support.
print("Starting camera feed...")

# Use CAP_DSHOW backend on Windows for faster startup
# CAP_DSHOW is more reliable and faster than the default backend on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Warm up the camera by capturing and discarding initial frames
# This helps stabilize the camera and improves startup performance
print("Warming up camera...")
for _ in range(5):
    cap.read()

print("Camera started. Running detection loop (press 'q' or ESC to stop)...")

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

# Function to draw bounding boxes on frame
def draw_bounding_boxes(frame, boxes):
    """
    Draws bounding boxes on the frame for detected objects.
    
    Args:
        frame: The frame to draw on
        boxes: YOLO detection boxes
        
    Returns:
        Frame with bounding boxes drawn
    """
    if not config.get('draw_bounding_boxes', True):
        return frame
    
    frame_with_boxes = frame.copy()
    thickness = config.get('bbox_thickness', 2)
    plate_color = tuple(config.get('plate_bbox_color', [0, 255, 0]))
    char_color = tuple(config.get('character_bbox_color', [255, 0, 0]))
    
    for box in boxes:
        # Get box coordinates (xyxy format)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Get class name and confidence
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]
        confidence = box.conf[0].item()
        
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
# Keep track of the latest frame with boxes for recording and display
latest_frame_with_boxes = None

# Create window for video display with resizable option
cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)

# Get screen dimensions and set initial window size based on config
try:
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    # Calculate window size based on percentage from config
    window_width_percent = config.get('window_width_percent', 80)
    window_height_percent = config.get('window_height_percent', 80)
    
    window_width = int(screen_width * window_width_percent / 100)
    window_height = int(screen_height * window_height_percent / 100)
    
    # Set window size
    cv2.resizeWindow('License Plate Detection', window_width, window_height)
    
    # Set fullscreen if configured
    if config.get('enable_fullscreen', False):
        cv2.setWindowProperty('License Plate Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print(f"Display window initialized: {window_width}x{window_height} ({window_width_percent}% x {window_height_percent}% of screen)")
except Exception as e:
    print(f"Could not set window size automatically: {e}")
    print("Window will use default size (you can still resize manually)")

# Track fullscreen state for toggle
is_fullscreen = config.get('enable_fullscreen', False)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available and draw
        try:
            plate_string, boxes = result_queue.get_nowait()
            
            # Draw bounding boxes on the frame
            latest_frame_with_boxes = draw_bounding_boxes(frame, boxes)
            
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
            # No new result, use the last frame with boxes if available
            if latest_frame_with_boxes is None:
                latest_frame_with_boxes = frame
        
        # Display the frame with bounding boxes
        if latest_frame_with_boxes is not None:
            # Add detected plate text to the display
            display_frame = latest_frame_with_boxes.copy()
            if last_printed_bottom:
                # Add text overlay with detected plate
                text = f"Detected: {last_printed_bottom}"
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('License Plate Detection', display_frame)
        
        # Write frame to video if recording is enabled
        if video_writer is not None and latest_frame_with_boxes is not None:
            video_writer.write(latest_frame_with_boxes)
        
        # Check for key press to exit (q or ESC) or toggle fullscreen (f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC key
            print("\nExiting...")
            break
        elif key == ord('f'):  # Toggle fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty('License Plate Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen mode enabled")
            else:
                cv2.setWindowProperty('License Plate Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen mode disabled")

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")

# --- 6. CLEANUP ---
# Performs necessary cleanup operations when the script is terminated.
# Ensures resources are properly released.
frame_queue.put(None) # Signal worker to exit
cap.release()
cv2.destroyAllWindows()

# Release video writer if it was initialized
if video_writer is not None:
    video_writer.release()
    print(f"Video recording saved successfully.")

print("Script finished.")
