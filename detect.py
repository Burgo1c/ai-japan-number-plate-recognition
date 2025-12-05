"""
License Plate Detection System (NCNN + Native Picamera2 Optimized)
"""

import cv2
import numpy as np
import time
import ncnn
import threading
import queue
import yaml
import os
import sys
from datetime import datetime
import csv

# --- 1. CONFIGURATION ---
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
INPUT_SIZE = 640

# --- DETERMINE ABSOLUTE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

# --- Load Configuration ---
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        raw_path = config['model_path']
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(BASE_DIR, raw_path)
            
        data_yaml_rel = config['data_yaml_path']
        if not os.path.isabs(data_yaml_rel):
            DATA_YAML_PATH = os.path.join(BASE_DIR, data_yaml_rel)
        else:
            DATA_YAML_PATH = data_yaml_rel
            
        # Default to USB if not specified, but user can set 'picamera' in config
        CAMERA_TYPE = config.get('camera_type', 'usb') # Options: 'usb', 'picamera'

except FileNotFoundError:
    print(f"Error: 'config.yaml' not found at {CONFIG_PATH}")
    sys.exit()
except Exception as e:
    print(f"Error loading config.yaml: {e}")
    sys.exit()

# --- Load Class Names ---
try:
    with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names']
        print(f"Loaded {len(class_names)} classes.")
except Exception as e:
    print(f"Error loading {DATA_YAML_PATH}: {e}")
    sys.exit()

# --- 2. CAMERA CLASS (Picamera2 + OpenCV Wrapper) ---
class CameraStream:
    def __init__(self, cam_type='usb', width=1280, height=720):
        self.type = cam_type
        self.width = width
        self.height = height
        self.cap = None
        self.picam2 = None
        
        if self.type == 'picamera':
            try:
                from picamera2 import Picamera2
                print("Initializing Native Picamera2...")
                self.picam2 = Picamera2()
                
                # Configure for Video (High FPS)
                # We use XRGB8888 because it aligns best with the Pi's hardware ISP
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "XRGB8888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                print(f"Picamera2 started at {self.width}x{self.height}")
            except ImportError:
                print("Error: 'picamera2' library not found. Install with: sudo apt install python3-picamera2")
                print("Falling back to USB mode.")
                self.type = 'usb'
            except Exception as e:
                print(f"Picamera2 Init Failed: {e}. Falling back to USB.")
                self.type = 'usb'

        if self.type == 'usb':
            print("Initializing USB Camera (V4L2)...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.type == 'picamera':
            # Capture array from Picamera2 (non-blocking if threaded properly, but here we poll)
            # This returns a generic array, usually (H, W, 4) for XRGB
            try:
                frame = self.picam2.capture_array()
                # Picamera2 returns RGB(A). OpenCV needs BGR.
                # Remove Alpha channel and Convert RGB to BGR
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            except Exception as e:
                print(f"Picamera error: {e}")
                return False, None
                
        else:
            return self.cap.read()

    def release(self):
        if self.type == 'picamera' and self.picam2:
            self.picam2.stop()
            self.picam2.close()
        elif self.cap:
            self.cap.release()

# --- 3. INITIALIZE NCNN MODEL ---
print(f"Looking for NCNN model...")
possible_paths = [
    raw_path,
    os.path.join(os.path.dirname(raw_path), 'best_ncnn_model'),
    os.path.join(BASE_DIR, 'best_ncnn_model'),
]

ncnn_model_dir = None
for p in possible_paths:
    if os.path.exists(p) and os.path.exists(os.path.join(p, 'model.ncnn.param')):
        ncnn_model_dir = p
        break

if not ncnn_model_dir:
    print("Error: Could not find 'best_ncnn_model' folder.")
    sys.exit()

param_path = os.path.join(ncnn_model_dir, 'model.ncnn.param')
bin_path = os.path.join(ncnn_model_dir, 'model.ncnn.bin')

try:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False 
    net.load_param(param_path)
    net.load_model(bin_path)
    print(f"Loaded NCNN model from: {ncnn_model_dir}")
except Exception as e:
    print(f"Error loading NCNN files: {e}")
    sys.exit()

# --- 4. INFERENCE & PROCESSING FUNCTIONS ---
def preprocess_image(image, input_size=640):
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    chw = np.ascontiguousarray(chw)
    return chw, scale, pad_w, pad_h

def postprocess_output(output, scale, pad_w, pad_h, conf_threshold=0.5):
    detections = []
    if len(output.shape) == 3: output = np.squeeze(output)
    if output.shape[0] == (len(class_names) + 4): output = output.transpose() 
    
    for detection in output:
        class_scores = detection[4:]
        max_conf = np.max(class_scores)
        
        if max_conf >= conf_threshold:
            class_id = np.argmax(class_scores)
            if class_id >= len(class_names): continue
            
            x_center, y_center, width, height = detection[0:4]
            x_center = (x_center - pad_w) / scale
            y_center = (y_center - pad_h) / scale
            w_original = width / scale
            h_original = height / scale
            
            x1 = x_center - w_original / 2
            y1 = y_center - h_original / 2
            x2 = x_center + w_original / 2
            y2 = y_center + h_original / 2
            
            detections.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "x": x_center, "y": y_center,
                "class": class_names[class_id],
                "confidence": float(max_conf)
            })
    return detections

def ncnn_inference(frame):
    input_data, scale, pad_w, pad_h = preprocess_image(frame, INPUT_SIZE)
    mat_in = ncnn.Mat(input_data)
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    ret, mat_out = ex.extract("out0")
    if ret != 0: return []
    output = np.array(mat_out)
    return postprocess_output(output, scale, pad_w, pad_h, CONF_THRESHOLD)

def get_yolo_parsed_strings(predictions):
    try:
        plate_detection = next(p for p in predictions if p["class"] == "NumberPLATE")
        divider_y = plate_detection["y"]
    except StopIteration:
        return None

    bottom_row = sorted([p for p in predictions if p["class"] != "NumberPLATE" and p["y"] >= divider_y], key=lambda p: p["x"])
    
    hiragana_char = ""
    number_string = ""
    for p in bottom_row:
        if p["class"].isdigit(): number_string += p["class"]
        else:
            if config['use_hiragana']: hiragana_char = config['location_dictionary'].get(p["class"], p["class"])
            else: hiragana_char = ""
    
    bottom_string = f"{hiragana_char}{number_string}"
    
    if config['get_top_line']:
        top_row = sorted([p for p in predictions if p["class"] != "NumberPLATE" and p["y"] < divider_y], key=lambda p: p["x"])
        top_string = ""
        for p in top_row:
            if p["class"].isdigit(): top_string += p["class"]
            else: top_string += config['location_dictionary'].get(p["class"], p["class"])
        if top_string: return f"{top_string}{bottom_string}"
    
    return bottom_string

def draw_bounding_boxes(frame, detections):
    if not config.get('draw_bounding_boxes', True): return frame
    
    frame_with_boxes = frame.copy()
    thickness = config.get('bbox_thickness', 2)
    plate_color = tuple(config.get('plate_bbox_color', [0, 255, 0]))
    char_color = tuple(config.get('character_bbox_color', [255, 0, 0]))
    
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(FRAME_WIDTH, x2), min(FRAME_HEIGHT, y2)
        
        color = plate_color if det['class'] == "NumberPLATE" else char_color
        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{det['class']} {det['confidence']:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_with_boxes, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
    return frame_with_boxes

# --- 5. WORKER THREAD ---
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

def worker():
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty: continue
        if frame is None: break

        detections = ncnn_inference(frame)
        plate_text = get_yolo_parsed_strings(detections)
        final_bottom = plate_text if plate_text else ""
        result_queue.put((final_bottom, detections))

# --- 6. MAIN LOOP ---

# Initialize Camera wrapper
# It will check config.yaml for 'camera_type'. If 'picamera', it loads the native lib.
camera_type = config.get('camera_type', 'usb') # 'usb' or 'picamera'
camera_stream = CameraStream(cam_type=camera_type, width=FRAME_WIDTH, height=FRAME_HEIGHT)

threading.Thread(target=worker, daemon=True).start()

# --- Init Recording/Logging ---
video_writer = None
if config.get('enable_recording', False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.get('output_directory', 'recordings')
    if not os.path.isabs(output_dir): output_dir = os.path.join(BASE_DIR, output_dir)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    video_filename = os.path.join(output_dir, f"recording_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
    print(f"Recording to: {video_filename}")

csv_file = None
csv_writer = None
if config.get('enable_csv_logging', False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = config.get('csv_output_directory', 'logs')
    if not os.path.isabs(csv_dir): csv_dir = os.path.join(BASE_DIR, csv_dir)
    if not os.path.exists(csv_dir): os.makedirs(csv_dir)
    
    csv_filename = os.path.join(csv_dir, f"detections_{timestamp}.csv")
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Plate Number'])
    print(f"Logging to: {csv_filename}")

print("Starting video... Press Ctrl+C to exit.")
last_printed_bottom = ""
latest_frame_with_boxes = None

try:
    while True:
        # Read from unified Camera Stream
        ret, frame = camera_stream.read()
        if not ret: 
            print("Failed to read frame")
            time.sleep(0.1)
            continue

        if frame_queue.empty():
            frame_queue.put(frame.copy())

        try:
            plate_string, detections = result_queue.get_nowait()
            latest_frame_with_boxes = draw_bounding_boxes(frame, detections)
            
            if plate_string and plate_string != last_printed_bottom:
                last_printed_bottom = plate_string
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{ts}] Detected: {last_printed_bottom}")
                
                if csv_writer:
                    csv_writer.writerow([ts, last_printed_bottom])
                    csv_file.flush()
            
        except queue.Empty:
            if latest_frame_with_boxes is None:
                latest_frame_with_boxes = frame
        
        if video_writer and latest_frame_with_boxes is not None:
            video_writer.write(latest_frame_with_boxes)

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")
finally:
    frame_queue.put(None)
    camera_stream.release()
    if video_writer: video_writer.release()
    if csv_file: csv_file.close()
    cv2.destroyAllWindows()
    print("Cleanup complete.")