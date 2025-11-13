import cv2
import numpy as np
import time
import easyocr
import yaml

# Try to import the TFLite runtime, which is efficient for Raspberry Pi
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("tflite_runtime not found. Falling back to tensorflow.lite.")
    print("For Raspberry Pi, install with: pip install tflite-runtime")
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: TensorFlow Lite is not installed.")
        print("Please install it: pip install tensorflow")
        exit()

# --- 1. CONFIGURATION ---
# !!! UPDATE THESE PATHS !!!
MODEL_PATH = '/home/raspi/ai-japan-number-plate-recognition/ai-model/best_int8.tflite' # <-- CHANGED
YAML_PATH = '/home/raspi/ai-japan-number-plate-recognition/ai-model/data.yaml'     # <-- ADDED
IMAGE_PATH = '/home/raspi/ai-japan-number-plate-recognition/test/plate1.jpg'
CONF_THRESHOLD = 0.5

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
    "nara": "奈良", "narasino": "習志野", "narita": "成田", "nasu": "那須", "ne": "ね", "nerima": "練馬", "ni": "に", "nigata": "新潟", "no": "の",
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

# --- Load Class Names (Required for TFLite) ---
try:
    with open(YAML_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']
    print("Loaded class names from data.yaml.")
except FileNotFoundError:
    print(f"Error: {YAML_PATH} not found.")
    exit()
except Exception as e:
    print(f"Error loading {YAML_PATH}: {e}")
    exit()

# --- Load TFLite Model ---
print("Loading TFLite model...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- ADD THIS DEBUG CODE ---
    print("\n--- TFLite Output Details (DEBUG) ---")
    for i, detail in enumerate(output_details):
        print(f"Output index {i}: Name={detail['name']}, Shape={detail['shape']}, Dtype={detail['dtype']}")
    print("---------------------------------------\n")
    # --- END DEBUG CODE ---

    # Get input shape
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    print("TFLite model loaded.")
except Exception as e:
    print(f"Error loading TFLite model from {MODEL_PATH}: {e}")
    exit()

# --- Load EasyOCR Model ---
print("Loading EasyOCR model...")
reader = easyocr.Reader(['ja', 'en'])
print("EasyOCR model loaded.")


# --- 3. LOAD IMAGE AND RUN PREDICTION ---
print(f"Loading image: {IMAGE_PATH}")
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("Error: Could not read image.")
    exit()

print("Running TFLite detection...")
start_time = time.time()

# --- TFLite Pre-processing ---
imH, imW, _ = frame.shape
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

# Check input type. INT8 models usually expect uint8
if input_details[0]['dtype'] == np.float32:
    # Normalize if model expects float
    input_data = (input_data.astype(np.float32) / 255.0)
# If input_details[0]['dtype'] is np.uint8, input_data is already correct

# --- TFLite Inference ---
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# --- TFLite Post-processing ---
# This assumes the TFLite model includes the NMS (Non-Max Suppression) op,
# which gives formatted outputs for boxes, classes, and scores.
boxes = interpreter.get_tensor(output_details[0]['index'])[0]   # [ymin, xmin, ymax, xmax]
classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class indices
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidences

# --- 4. PARSE AND COMBINE RESULTS ---

# This function is the same, but it will be fed by our TFLite loop
def get_yolo_parsed_strings(predictions):
    try:
        plate_detection = next(p for p in predictions if p["class"] == "NumberPLATE")
        divider_y = plate_detection["y"]
    except StopIteration:
        return None, None

    top_row_detections = sorted([p for p in predictions if p["class"] != "NumberPLATE" and p["y"] < divider_y], key=lambda p: p["x"])
    bottom_row_detections = sorted([p for p in predictions if p["class"] != "NumberPLATE" and p["y"] >= divider_y], key=lambda p: p["x"])

    top_string = "".join([location_dictionary.get(p["class"], p["class"]) for p in top_row_detections])
    
    hiragana_char = ""
    number_string = ""
    for p in bottom_row_detections:
        if p["class"].isdigit():
            number_string += p["class"]
        else:
            hiragana_char = location_dictionary.get(p["class"], p["class"])

    if len(number_string) == 4:
        number_string = f"{number_string[:2]}-{number_string[2:]}"
    
    bottom_string = f"{hiragana_char} {number_string}"
    return top_string, bottom_string

# --- Process TFLite results ---
yolo_predictions = []       # For the parser
detections_for_drawing = [] # For the visualizer
plate_box_for_ocr = None    # For the OCR

for i in range(len(scores)):
    if ((scores[i] > CONF_THRESHOLD) and (scores[i] <= 1.0)):
        # Get bounding box coordinates
        ymin = int(max(1, (boxes[i][0] * imH)))
        xmin = int(max(1, (boxes[i][1] * imW)))
        ymax = int(min(imH, (boxes[i][2] * imH)))
        xmax = int(min(imW, (boxes[i][3] * imW)))
        
        class_id = int(classes[i])
        class_name = class_names[class_id]

        # For parser (needs center x, y)
        yolo_predictions.append({
            "x": (xmin + xmax) / 2,
            "y": (ymin + ymax) / 2,
            "class": class_name,
        })

        # For drawing
        detections_for_drawing.append({
            "box": [xmin, ymin, xmax, ymax],
            "class_name": class_name,
            "confidence": scores[i]
        })

        if class_name == "NumberPLATE":
            plate_box_for_ocr = [xmin, ymin, xmax, ymax]

end_time = time.time()
print(f"Finished TFLite detection in {end_time - start_time:.2f} seconds.")

# --- Parse the TFLite string predictions ---
yolo_top, yolo_bottom = get_yolo_parsed_strings(yolo_predictions)

# --- Process OCR results ---
ocr_top = ""
if plate_box_for_ocr is not None:
    x1, y1, x2, y2 = plate_box_for_ocr
    plate_img = frame[y1:y2, x1:x2]
    
    print("Running OCR on the license plate...")
    ocr_result = reader.readtext(plate_img, detail=0, paragraph=True)
    if ocr_result:
        full_ocr_text = " ".join(ocr_result)
        parts = full_ocr_text.split()
        for part in parts:
            if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9faf' for char in part):
                ocr_top = "".join(c for c in part if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
                break

# --- 5. COMBINE AND DISPLAY ---
final_top = ocr_top if ocr_top else yolo_top
final_bottom = yolo_bottom if yolo_bottom else ""

print("\n--- Combined License Plate ---")
print(f"Top Row:    {final_top} {yolo_top if yolo_top and ocr_top else ''}")
print(f"Bottom Row: {final_bottom}")
print(f"Full Plate: {final_top} {yolo_top if yolo_top and ocr_top else ''} {final_bottom}\n")

# --- Visualize (using detections_for_drawing) ---
for det in detections_for_drawing:
    x1, y1, x2, y2 = det["box"]
    class_name = det["class_name"]
    confidence = det["confidence"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
