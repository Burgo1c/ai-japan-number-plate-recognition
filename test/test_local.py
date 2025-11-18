import cv2
import numpy as np
import time
from ultralytics import YOLO
import easyocr

# --- 1. CONFIGURATION ---
MODEL_PATH = 'ai-model/best.pt'
IMAGE_PATH = 'test/plate1.jpg'
CONF_THRESHOLD = 0.5

# Location Dictionary
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
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded.")
print("Loading EasyOCR model...")
reader = easyocr.Reader(['ja', 'en'])
print("EasyOCR model loaded.")

# --- 3. LOAD IMAGE AND RUN PREDICTION ---
print(f"Loading image: {IMAGE_PATH}")
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("Error: Could not read image.")
    exit()

print("Running YOLO detection...")
start_time = time.time()
yolo_results = model.predict(source=frame, conf=CONF_THRESHOLD, save=False)
end_time = time.time()
print(f"Finished YOLO detection in {end_time - start_time:.2f} seconds.")

# --- 4. PARSE AND COMBINE RESULTS ---
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

# Process YOLO results
yolo_predictions = []
for box in yolo_results[0].boxes:
    yolo_predictions.append({
        "x": box.xywh[0][0].item(),
        "y": box.xywh[0][1].item(),
        "class": model.names[int(box.cls[0].item())],
    })

yolo_top, yolo_bottom = get_yolo_parsed_strings(yolo_predictions)

# Process OCR results
ocr_top = ""
plate_box = next((box for box in yolo_results[0].boxes if model.names[int(box.cls[0].item())] == "NumberPLATE"), None)

if plate_box is not None:
    x1, y1, x2, y2 = plate_box.xyxy[0].int().tolist()
    plate_img = frame[y1:y2, x1:x2]
    
    print("Running OCR on the license plate...")
    ocr_result = reader.readtext(plate_img, detail=0, paragraph=True)
    if ocr_result:
        # Assume the first line of OCR result contains the location
        full_ocr_text = " ".join(ocr_result)
        # Heuristic: Find the first non-numeric/non-latin part as the location
        parts = full_ocr_text.split()
        for part in parts:
            # A simple check for Japanese characters
            if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9faf' for char in part):
                # Filter to keep only Japanese characters
                ocr_top = "".join(c for c in part if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf')
                break

# --- 5. COMBINE AND DISPLAY ---
final_top = ocr_top if ocr_top else yolo_top
final_bottom = yolo_bottom if yolo_bottom else ""

print("\n--- Combined License Plate ---")
print(f"Top Row:    {final_top} {yolo_top if yolo_top and ocr_top else ''}")
print(f"Bottom Row: {final_bottom}")
print(f"Full Plate: {final_top} {yolo_top if yolo_top and ocr_top else ''} {final_bottom}\n")

# Visualize
for box in yolo_results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
    class_name = model.names[int(box.cls[0].item())]
    confidence = box.conf[0].item()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
