import cv2
import easyocr
from ultralytics import YOLO
from picamera2 import Picamera2
import time

# --- 1. CONFIGURATION ---
MODEL_PATH = 'best.pt' # Your trained YOLO model
JAPANESE_ALLOWLIST = '天地人あいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわをん品川足立練馬多摩横浜川崎湘南相模札幌函館旭川室蘭釧路帯広北見大阪なにわ和泉01234456789·-'

# --- 2. HELPER FUNCTION ---
def preprocess_for_ocr(plate_image):
    """Applies grayscale and adaptive thresholding to clean the image."""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# --- 3. INITIALIZE MODELS (Do this once) ---
print("Loading models... This will be slow.")

# Load YOLO model (on CPU)
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Load EasyOCR model (on CPU)
try:
    ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
    print("EasyOCR model loaded.")
except Exception as e:
    print(f"Error loading EasyOCR: {e}")
    exit()

# --- 4. INITIALIZE CAMERA ---
try:
    picam2 = Picamera2()
    # Configure for a high-res still image
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    print("Camera initialized. Waiting 2s for warmup...")
    time.sleep(2.0) # Give camera time to adjust exposure
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit()

# --- 5. CAPTURE AND PROCESS ---
print("Capturing image...")
try:
    # picamera2 captures in RGB, OpenCV uses BGR
    # We capture the array and convert the color space
    rgb_frame = picam2.capture_array("main")
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    print("Image captured. Running detection...")
    start_time = time.time()
    
    # 1. Detect all plates
    results = model(frame, verbose=False) 
    plates_found = 0

    # 2. Loop through all detections
    for result in results:
        for box in result.boxes:
            
            # 3. Crop Plate
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue

            # 4. Clean Image for OCR
            clean_plate = preprocess_for_ocr(plate_crop)
            
            # 5. Read Text (EasyOCR)
            ocr_result = ocr_reader.readtext(
                clean_plate, 
                detail=0, 
                paragraph=True,
                allowlist=JAPANESE_ALLOWLIST
            )
            
            if ocr_result:
                plates_found += 1
                plate_text = "".join(ocr_result).replace(" ", "")
                # --- GOAL ACHIEVED ---
                print(f"--- PLATE {plates_found} DETECTED: {plate_text} ---")

    end_time = time.time()
    print(f"Finished processing in {end_time - start_time:.2f} seconds.")
    if plates_found == 0:
        print("No plates found in the image.")

finally:
    # 6. Shutdown camera
    picam2.stop()
    print("Camera stopped.")