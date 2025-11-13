import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
import queue

# --- 1. CONFIGURATION ---
MODEL_PATH = 'ai-model/best.pt'
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- 2. INITIALIZE MODELS ---
print("Loading YOLO model (best.pt)...")
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure 'ultralytics' is installed (pip install ultralytics)")
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

    number_string = ""
    for p in bottom_row_detections:
        if p["class"].isdigit():
            number_string += p["class"]

    if len(number_string) == 4:
        number_string = f"{number_string[:2]}-{number_string[2:]}"
    
    return number_string

# --- 4. WORKER THREAD FOR PROCESSING ---
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
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
                "class": model.names[int(box.cls[0].item())],
            })

        yolo_bottom = get_yolo_parsed_strings(yolo_predictions)

        # --- Combine and Pass Results ---
        final_bottom = yolo_bottom if yolo_bottom else ""

        # Pass all results to the main thread
        result_queue.put((final_bottom, yolo_results[0].boxes))

# --- 5. MAIN THREAD FOR CAMERA AND CONSOLE ---
print("Starting camera feed...")
# Note: On Raspberry Pi, camera index 0 might be the CSI camera or a USB camera.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    print("On Raspberry Pi, ensure the camera is enabled (sudo raspi-config)")
    print("and that OpenCV has permission or the correct backend.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
print("Camera started. Running detection loop (press Ctrl+C to stop)...")

threading.Thread(target=worker, daemon=True).start()

# Keep track of the last printed result to avoid spam
last_printed_top = ""
last_printed_bottom = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame. Exiting.")
            break

        # Put frame in queue for processing if the worker is ready
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get results from the queue if available and draw
        try:
            final_bottom, boxes = result_queue.get_nowait()
            
            # --- CONSOLE OUTPUT ---
            # Ignoring top row as per user request to focus on bottom row.
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
cap.release()
print("Script finished.")
