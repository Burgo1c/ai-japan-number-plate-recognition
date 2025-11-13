import cv2
import numpy as np
import time
import threading
import queue
import easyocr
import yaml
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
MODEL_PATH = '/home/raspi/ai-japan-number-plate-recognition/ai-model/best.pt'
CONF_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Load Class Names ---
try:
    with open('/home/raspi/ai-japan-number-plate-recognition/ai-model/data.yaml', 'r') as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']
except FileNotFoundError:
    print("Error: '/home/raspi/ai-japan-number-plate-recognition/ai-model/data.yaml' not found.")
    print("Please ensure the data.yaml file is in the correct location.")
    exit()
except Exception as e:
    print(f"Error loading data.yaml: {e}")
    exit()


# Location Dictionary (remains the same)
location_dictionary = {
    "a": "あ", "adati": "足立", "aidu": "会津", "akita": "秋田", "amami": "奄美", "aomori": "青森", "asahikawa": "旭川", "asuka": "飛鳥", "e": "え", "ehime": "愛媛",
    "gihu": "岐阜", "gunma": "群馬", "ha": "は", "hakodate": "函館", "hamamatu": "浜松", "hatinohe": "八戸", "hatiouzi": "八王子", "he": "へ", "hi": "ひ",
    "hida": "飛騨", "himezi": "姫路", "hiraizumi": "平泉", "hirosaki": "弘前", "hiroshima": "広島", "ho": "ほ", "hu": "ふ", "hukui": "福井",
    "hukuoka": "福岡", "hukushima": "福島", "hukuyama": "福山", "hunabashi": "船橋", "huzisan": "富士山", "i": "い", "isesima": "伊勢志摩",
    "isikawa": "石川", "itabashi": "板橋", "itihara": "市原", "itikawa": "市川", "itinomiya": "一宮", "iwaki": "いわき", "iwate": "岩手", "izu": "伊豆",
    "izumi": "和泉", "izumo": "出雲", "ka": "か", "kagawa": "香川", "kagosima": "鹿児島", "kanazawa": "金沢", "kasiwa": "柏", "kasugai": "春日井",
    "kasukabe": "春日部", "katusika": "葛飾", "kawagoe": "川越", "kawaguti": "川口", "kawasaki": "川崎", "ke": "け", "ki": "き", "kitakyusyu": "北九州",
    "kitami": "北見", "ko": "こ", "koube": "神戸", "kouriyama": "郡山", "kouti": "高知", "kouto": "江東", "ku": "く", "kumagaya": "熊谷",
    "kumamoto": "熊本", "kurasiki": "倉敷", "kusiro": "釧路", "kyoto": "京都", "ma": "ま", "maebashi": "前橋", "matudo": "松戸", "matumoto": "松本",
    "me": "め", "mi": "み", "mie": "三重", "mikawa": "三河", "mito": "水戸", "miyagi": "宮城", "miyazaki": "宮崎", "mo": "も", "morioka": "盛岡",
    "mu": "む", "muroran": "室蘭", "na": "な", "nagano": "長野", "nagaoka": "長岡", "nagasaki": "長崎", "nagoya": "名古屋", "naniwa": "なにわ",
    "nara": "奈良", "narasino": "習志னோ", "narita": "成田", "nasu": "那須", "ne": "ね", "nerima": "練馬", "ni": "に", "nigata": "新潟", "no": "の",
    "noda": "野田", "nu": "ぬ", "numadu": "沼津", "o": "お", "obihiro": "帯広", "oita": "大分", "okayama": "岡山", "okazaki": "岡崎", "okinawa": "沖縄",
    "omiya": "大宮", "osaka": "大阪", "owarikomaki": "尾張小牧", "ra": "ら", "re": "れ", "ri": "り", "ro": "ろ", "ru": "る", "sa": "さ", "saga": "佐賀",
    "sagami": "相模", "sakai": "堺", "sasebo": "佐世保", "se": "せ", "sendai": "仙台", "setagaya": "世田谷", "si": "し", "siga": "滋賀",
    "simane": "島根", "simonoseki": "下関", "sinagawa": "品川", "sirakawa": "白河", "siretoko": "知床", "sizuoka": "静岡", "so": "そ",
    "sodegaura": "袖ヶ浦", "su": "す", "suginami": "杉並", "suwa": "諏訪", "suzuka": "鈴鹿", "syounai": "庄内", "syounan": "湘南", "ta": "た",
    "takamatu": "高松", "takasaki": "高崎", "tama": "多摩", "te": "て", "ti": "ち", "tiba": "千葉", "tikuho": "筑豊", "to": "と", "tokorozawa": "所沢",
    "tokusima": "徳島", "tomakomaki": "苫小牧", "totigi": "栃木", "tottori": "鳥取", "toyama": "富山", "toyohasi": "豊橋", "toyota": "豊田", "tu": "つ",
    "tukuba": "つくば", "tutiura": "土浦", "u": "う", "utunomiya": "宇都宮", "wa": "わ", "wo": "を", "ya": "や", "yamagata": "山形", "yamaguti": "山口",
    "yamanashi": "山梨", "yo": "よ", "yokkaiti": "四日市", "yokohama": "横浜", "yu": "ゆ", "zyoetu": "上越"
}

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


print("Loading EasyOCR model...")
try:
    reader = easyocr.Reader(['ja', 'en'])
    print("EasyOCR model loaded.")
except Exception as e:
    print(f"Error loading EasyOCR: {e}")
    exit()

# --- 3. PARSING FUNCTION ---
def get_yolo_parsed_strings(plate_detection, char_detections):
    """
    Parses character detections relative to the plate detection.
    """
    if not plate_detection or not char_detections:
        return None, None

    # Use the plate's center Y as the divider
    plate_box = plate_detection['box']
    divider_y = (plate_box[1] + plate_box[3]) / 2

    top_row_detections = sorted(
        [p for p in char_detections if (p['box'][1] + p['box'][3]) / 2 < divider_y],
        key=lambda p: p['box'][0]
    )
    bottom_row_detections = sorted(
        [p for p in char_detections if (p['box'][1] + p['box'][3]) / 2 >= divider_y],
        key=lambda p: p['box'][0]
    )

    # --- Build Strings ---
    top_string = "".join([location_dictionary.get(p["class_name"], p["class_name"]) for p in top_row_detections])
    
    hiragana_char = ""
    number_string = ""
    for p in bottom_row_detections:
        if p["class_name"].isdigit():
            number_string += p["class_name"]
        else:
            hiragana_char = location_dictionary.get(p["class_name"], p["class_name"])

    if len(number_string) == 4:
        number_string = f"{number_string[:2]}-{number_string[2:]}"
    
    bottom_string = f"{hiragana_char} {number_string}".strip()
    return top_string, bottom_string

# --- 4. WORKER THREAD FOR PROCESSING ---
frame_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # --- YOLO Inference ---
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        detections = []       # For drawing (not used here, but good for debug)
        char_detections = []  # For the parser
        plate_box_dict = None # For OCR and parser

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].cpu().numpy()]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = class_names[cls_id]

            detection_dict = {
                "box": [x1, y1, x2, y2],
                "class_name": class_name,
                "confidence": conf
            }
            
            detections.append(detection_dict) 

            if class_name == "NumberPLATE":
                plate_box_dict = detection_dict
            else:
                char_detections.append(detection_dict)

        # --- OCR and Parsing ---
        ocr_text = ""
        yolo_top, yolo_bottom = None, None
        
        if plate_box_dict:
            x1, y1, x2, y2 = plate_box_dict["box"]
            padding = 5
            imH, imW, _ = frame.shape
            y1_pad = max(0, y1 - padding)
            y2_pad = min(imH, y2 + padding)
            x1_pad = max(0, x1 - padding)
            x2_pad = min(imW, x2 + padding)

            plate_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if plate_img.size > 0:
                try:
                    ocr_result = reader.readtext(plate_img, detail=0, paragraph=True)
                    if ocr_result:
                        ocr_text = " ".join(ocr_result)
                except Exception as e:
                    print(f"EasyOCR Error: {e}")
                    ocr_text = "OCR Error"

            yolo_top, yolo_bottom = get_yolo_parsed_strings(plate_box_dict, char_detections)

        # Pass all results to the main thread
        result_queue.put((detections, ocr_text, yolo_top, yolo_bottom))

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

        # Get results from the queue if available
        try:
            # We only care about the parsed strings for console output
            _, ocr_text, yolo_top, yolo_bottom = result_queue.get_nowait()
            
            # --- CONSOLE OUTPUT ---
            # Only print if a valid new result is found
            new_result = False
            if yolo_top and yolo_top != last_printed_top:
                last_printed_top = yolo_top
                new_result = True
                
            if yolo_bottom and yolo_bottom != last_printed_bottom:
                last_printed_bottom = yolo_bottom
                new_result = True

            if new_result:
                print("--- Plate Detected ---")
                print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Top:    {last_printed_top}")
                print(f"Bottom: {last_printed_bottom}")
                # Optional: Uncomment to print OCR result as well
                # print(f"OCR:    {ocr_text}")
                print("----------------------")
            
        except queue.Empty:
            pass # No new result, just keep looping
        
        # Give a tiny break to the CPU
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")

# --- 6. CLEANUP ---
frame_queue.put(None) # Signal worker to exit
cap.release()
print("Script finished.")