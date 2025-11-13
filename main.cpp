#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#include <condition_variable>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <yaml-cpp/yaml.h>
#include <curl/curl.h>
#include <sstream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

// --- 1. CONFIGURATION ---

// !!! IMPORTANT !!!
// Change this to the path of your NEW float16 model
const std::string MODEL_PATH = "/home/raspi/ai-japan-number-plate-recognition/ai-model/best_fp16.tflite";
const std::string DATA_YAML_PATH = "/home/raspi/ai-japan-number-plate-recognition/ai-model/data.yaml";

// Model parameters
const int IMG_SIZE = 640;
const float CONF_THRESHOLD = 0.5f;
const float IOU_THRESHOLD = 0.45f;

// --- Location Dictionary ---
std::map<std::string, std::string> location_dictionary = {
    {"a", "あ"}, {"adati", "足立"}, {"aidu", "会津"}, {"akita", "秋田"}, {"amami", "奄美"}, {"aomori", "青森"}, {"asahikawa", "旭川"}, {"asuka", "飛鳥"}, {"e", "え"}, {"ehime", "愛媛"},
    {"gihu", "岐阜"}, {"gunma", "群馬"}, {"ha", "は"}, {"hakodate", "函館"}, {"hamamatu", "浜松"}, {"hatinohe", "八戸"}, {"hatiouzi", "八王子"}, {"he", "へ"}, {"hi", "ひ"},
    {"hida", "飛騨"}, {"himezi", "姫路"}, {"hiraizumi", "平泉"}, {"hirosaki", "弘前"}, {"hiroshima", "広島"}, {"ho", "ほ"}, {"hu", "ふ"}, {"hukui", "福井"},
    {"hukuoka", "福岡"}, {"hukushima", "福島"}, {"hukuyama", "福山"}, {"hunabashi", "船橋"}, {"huzisan", "富士山"}, {"i", "い"}, {"isesima", "伊勢志摩"},
    {"isikawa", "石川"}, {"itabashi", "板橋"}, {"itihara", "市原"}, {"itikawa", "市川"}, {"itinomiya", "一宮"}, {"iwaki", "いわき"}, {"iwate", "岩手"}, {"izu", "伊豆"},
    {"izumi", "和泉"}, {"izumo", "出雲"}, {"ka", "か"}, {"kagawa", "香川"}, {"kagosima", "鹿児島"}, {"kanazawa", "金沢"}, {"kasiwa", "柏"}, {"kasugai", "春日井"},
    {"kasukabe", "春日部"}, {"katusika", "葛飾"}, {"kawagoe", "川越"}, {"kawaguti", "川口"}, {"kawasaki", "川崎"}, {"ke", "け"}, {"ki", "き"}, {"kitakyusyu", "北九州"},
    {"kitami", "北見"}, {"ko", "こ"}, {"koube", "神戸"}, {"kouriyama", "郡山"}, {"kouti", "高知"}, {"kouto", "江東"}, {"ku", "く"}, {"kumagaya", "熊谷"},
    {"kumamoto", "熊本"}, {"kurasiki", "倉敷"}, {"kusiro", "釧路"}, {"kyoto", "京都"}, {"ma", "ま"}, {"maebashi", "前橋"}, {"matudo", "松戸"}, {"matumoto", "松本"},
    {"me", "め"}, {"mi", "み"}, {"mie", "三重"}, {"mikawa", "三河"}, {"mito", "水戸"}, {"miyagi", "宮城"}, {"miyazaki", "宮崎"}, {"mo", "も"}, {"morioka", "盛岡"},
    {"mu", "む"}, {"muroran", "室蘭"}, {"na", "な"}, {"nagano", "長野"}, {"nagaoka", "長岡"}, {"nagasaki", "長崎"}, {"nagoya", "名古屋"}, {"naniwa", "なにわ"},
    {"nara", "奈良"}, {"narasino", "習志野"}, {"narita", "成田"}, {"nasu", "那須"}, {"ne", "ね"}, {"nerima", "練馬"}, {"ni", "に"}, {"nigata", "新潟"}, {"no", "の"},
    {"noda", "野田"}, {"nu", "ぬ"}, {"numadu", "沼津"}, {"o", "お"}, {"obihiro", "帯広"}, {"oita", "大分"}, {"okayama", "岡山"}, {"okazaki", "岡崎"}, {"okinawa", "沖縄"},
    {"omiya", "大宮"}, {"osaka", "大阪"}, {"owarikomaki", "尾張小牧"}, {"ra", "ら"}, {"re", "れ"}, {"ri", "り"}, {"ro", "ろ"}, {"ru", "る"}, {"sa", "さ"}, {"saga", "佐賀"},
    {"sagami", "相模"}, {"sakai", "堺"}, {"sasebo", "佐世保"}, {"se", "せ"}, {"sendai", "仙台"}, {"setagaya", "世田谷"}, {"si", "し"}, {"siga", "滋賀"},
    {"simane", "島根"}, {"simonoseki", "下関"}, {"sinagawa", "品川"}, {"sirakawa", "白河"}, {"siretoko", "知床"}, {"sizuoka", "静岡"}, {"so", "そ"},
    {"sodegaura", "袖ヶ浦"}, {"su", "す"}, {"suginami", "杉並"}, {"suwa", "諏訪"}, {"suzuka", "鈴鹿"}, {"syounai", "庄内"}, {"syounan", "湘南"}, {"ta", "た"},
    {"takamatu", "高松"}, {"takasaki", "高崎"}, {"tama", "多摩"}, {"te", "て"}, {"ti", "ち"}, {"tiba", "千葉"}, {"tikuho", "筑豊"}, {"to", "と"}, {"tokorozawa", "所沢"},
    {"tokusima", "徳島"}, {"tomakomaki", "苫小牧"}, {"totigi", "栃木"}, {"tottori", "鳥取"}, {"toyama", "富山"}, {"toyohasi", "豊橋"}, {"toyota", "豊田"}, {"tu", "つ"},
    {"tukuba", "つくば"}, {"tutiura", "土浦"}, {"u", "う"}, {"utunomiya", "宇都宮"}, {"wa", "わ"}, {"wo", "を"}, {"ya", "や"}, {"yamagata", "山形"}, {"yamaguti", "山口"},
    {"yamanashi", "山梨"}, {"yo", "よ"}, {"yokkaiti", "四日市"}, {"yokohama", "横浜"}, {"yu", "ゆ"}, {"zyoetu", "上越"}
};

struct Detection {
    float x, y, width, height, confidence;
    std::string class_name;
    int class_id;
};

// --- Globals for Queue and Cache ---
std::queue<std::string> plate_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
bool finished = false;

std::map<std::string, std::chrono::steady_clock::time_point> recent_plates;
const std::chrono::seconds CACHE_DURATION(30);

// --- Graceful Shutdown ---
volatile sig_atomic_t g_running = 1;
void signal_handler(int signum) {
    g_running = 0;
    std::cout << "\nCaught signal " << signum << ", shutting down..." << std::endl;
}


void send_to_api(const std::string& complete_plate) {
    CURL *curl = curl_easy_init();
    if(curl) {
        std::string api_url = "http://your.api.endpoint/plate"; // <-- IMPORTANT: CHANGE THIS
        std::string json_data = "{\"complete_number_plate\": \"" + complete_plate + "\"}";

        curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); 

        CURLcode res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            std::cerr << "API call failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Successfully sent plate to API: " << complete_plate << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

// --- Consumer Thread Function ---
void api_consumer_thread() {
    while (true) {
        std::string plate_to_send;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, []{ return !plate_queue.empty() || finished; });

            if (finished && plate_queue.empty()) {
                break;
            }

            plate_to_send = plate_queue.front();
            plate_queue.pop();
        } 

        // --- API CALL REMAINS COMMENTED AS REQUESTED ---
        //send_to_api(plate_to_send);
    }
    std::cout << "API consumer thread finished." << std::endl;
}

// --- OCR Function ---
std::string run_ocr(const cv::Mat& image) {
    if (image.empty()) {
        return "";
    }

    tesseract::TessBaseAPI ocr;
    if (ocr.Init(NULL, "jpn+eng", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Could not initialize tesseract." << std::endl;
        return "";
    }

    ocr.SetImage(image.data, image.cols, image.rows, image.channels(), image.step);
    
    char* outText = ocr.GetUTF8Text();
    std::string ocr_result(outText);
    
    ocr.End();
    delete[] outText;

    // Clean up the OCR result: remove spaces and newlines
    ocr_result.erase(std::remove_if(ocr_result.begin(), ocr_result.end(), ::isspace), ocr_result.end());

    return ocr_result;
}


// --- Helper function to check if a string is composed only of digits ---
bool is_digit(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

// --- 2. PARSING FUNCTION (REVISED) ---
std::pair<std::string, std::string> parse_plate_detections(const std::vector<Detection>& predictions, const std::string& ocr_text) {
    float divider_y = -1.0f;

    // Find the license plate and calculate its vertical center.
    for (const auto& p : predictions) {
        if (p.class_name == "NumberPLATE") {
            divider_y = p.y + p.height / 2.0f; // Center Y of the plate
            break;
        }
    }

    // If no plate is found, we can't proceed.
    if (divider_y == -1.0f) {
        return {"", ""};
    }

    std::vector<Detection> top_row_detections, bottom_row_detections;
    for (const auto& p : predictions) {
        if (p.class_name == "NumberPLATE") continue;
        
        // Use the character's vertical center for comparison.
        float char_center_y = p.y + p.height / 2.0f;
        if (char_center_y < divider_y) {
            top_row_detections.push_back(p);
        } else {
            bottom_row_detections.push_back(p);
        }
    }

    std::sort(top_row_detections.begin(), top_row_detections.end(), [](const Detection& a, const Detection& b) { return a.x < b.x; });
    std::sort(bottom_row_detections.begin(), bottom_row_detections.end(), [](const Detection& a, const Detection& b) { return a.x < b.x; });

    // --- Top Row Logic ---
    std::stringstream top_ss;
    for (const auto& p : top_row_detections) {
        auto it = location_dictionary.find(p.class_name);
        top_ss << (it != location_dictionary.end() ? it->second : p.class_name);
    }
    std::string yolo_top_string = top_ss.str();

    // --- NEW Bottom Row Logic (from Python) ---
    std::string hiragana_char;
    std::string number_string;
    for (const auto& p : bottom_row_detections) {
        if (is_digit(p.class_name)) {
            number_string += p.class_name;
        } else {
            // Assume any non-digit is the hiragana
            auto it = location_dictionary.find(p.class_name);
            hiragana_char = (it != location_dictionary.end() ? it->second : p.class_name);
        }
    }

    // Add hyphen for 4-digit numbers
    if (number_string.length() == 4) {
        number_string.insert(2, "-");
    }
    
    std::string bottom_string;
    if (!hiragana_char.empty()) {
        bottom_string += hiragana_char;
        if (!number_string.empty()) {
            bottom_string += " " + number_string;
        }
    } else {
        bottom_string = number_string;
    }
    // --- End of NEW Bottom Row Logic ---


    // Use OCR text for the top row if available, otherwise fall back to YOLO
    std::string final_top_string = ocr_text.empty() ? yolo_top_string : ocr_text;

    std::cout << "\n--- Parsed License Plate ---" << std::endl;
    std::cout << "OCR Top:      " << ocr_text << std::endl;
    std::cout << "YOLO Top:     " << yolo_top_string << std::endl;
    std::cout << "Bottom:       " << bottom_string << std::endl;
    std::cout << "Complete:     " << final_top_string << " " << bottom_string << std::endl;

    return {final_top_string, bottom_string};
}


int main() {
    signal(SIGINT, signal_handler);
    curl_global_init(CURL_GLOBAL_ALL);

    std::thread consumer_thread(api_consumer_thread);
    std::cout << "API consumer thread started." << std::endl;

    // --- 3. INITIALIZE MODEL & LOAD DATA ---
    std::cout << "Loading TFLite model..." << std::endl;
    std::vector<std::string> class_names;
    try {
        YAML::Node data = YAML::LoadFile(DATA_YAML_PATH);
        const YAML::Node& names_node = data["names"];
        for (const auto& name : names_node) {
            class_names.push_back(name.as<std::string>());
        }
        std::cout << "Loaded " << class_names.size() << " class names." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading data.yaml: " << e.what() << std::endl;
        return -1;
    }

    auto model = tflite::FlatBufferModel::BuildFromFile(MODEL_PATH.c_str());
    if (!model) {
        std::cerr << "Failed to load model " << MODEL_PATH << std::endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter." << std::endl;
        return -1;
    }

    interpreter->SetNumThreads(3);
    interpreter->AllocateTensors();

    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteType input_type = interpreter->tensor(input_tensor_idx)->type;
    std::cout << "Model loaded." << (input_type == kTfLiteFloat32 ? " (Float32)" : (input_type == kTfLiteFloat16 ? " (Float16)" : " (Int8/Other)")) << std::endl;


    // --- 4. WEBCAM CAPTURE AND CONTINUOUS DETECTION ---
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    std::cout << "Camera opened. Starting continuous detection..." << std::endl;
    std::cout << "Press 'q' to quit." << std::endl;

    cv::Mat frame;
    while (g_running) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame." << std::endl;
            break;
        }

        // --- Pre-processing ---
        int orig_h = frame.rows;
        int orig_w = frame.cols;
        float scale = std::min(static_cast<float>(IMG_SIZE) / orig_h, static_cast<float>(IMG_SIZE) / orig_w);
        int new_w = static_cast<int>(orig_w * scale);
        int new_h = static_cast<int>(orig_h * scale);

        cv::Mat resized_img;
        cv::resize(frame, resized_img, cv::Size(new_w, new_h));

        cv::Mat input_img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
        int top_pad = (IMG_SIZE - new_h) / 2;
        int left_pad = (IMG_SIZE - new_w) / 2;
        resized_img.copyTo(input_img(cv::Rect(left_pad, top_pad, new_w, new_h)));

        // --- Model Input Handling (Handles FP16/FP32/INT8) ---
        if (input_type == kTfLiteFloat32) {
            cv::Mat float_img;
            input_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
            memcpy(interpreter->typed_tensor<float>(input_tensor_idx), float_img.data, float_img.total() * float_img.elemSize());
        } else if (input_type == kTfLiteUInt8) {
            memcpy(interpreter->typed_tensor<uint8_t>(input_tensor_idx), input_img.data, input_img.total() * input_img.elemSize());
        } else {
             // Assuming FP16 is treated like FP32 for input data conversion
            cv::Mat float_img;
            input_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
            memcpy(interpreter->typed_tensor<float>(input_tensor_idx), float_img.data, float_img.total() * float_img.elemSize());
        }


        interpreter->Invoke();

        const float* output_data = interpreter->typed_output_tensor<float>(0);
        TfLiteIntArray* output_dims = interpreter->tensor(interpreter->outputs()[0])->dims;
        const int num_proposals = output_dims->data[2];
        const int num_classes = output_dims->data[1] - 4;

        // --- START NMS FIX ---
        std::vector<cv::Rect2f> char_boxes_for_nms;
        std::vector<float> char_confidences;
        std::vector<int> char_class_ids;
        std::vector<Detection> plate_detections; 

        for (int i = 0; i < num_proposals; ++i) {
            float max_conf = 0.0f;
            int class_id = -1;
            for (int j = 0; j < num_classes; ++j) {
                float conf = output_data[i + (4 + j) * num_proposals];
                if (conf > max_conf) {
                    max_conf = conf;
                    class_id = j;
                }
            }

            if (max_conf > CONF_THRESHOLD) {
                std::string class_name = class_names[class_id];
                
                float cx = output_data[i + 0 * num_proposals];
                float cy = output_data[i + 1 * num_proposals];
                float w = output_data[i + 2 * num_proposals];
                float h = output_data[i + 3 * num_proposals];
                
                if (class_name == "NumberPLATE") {
                    Detection det;
                    det.x = (cx - w / 2 - left_pad) / scale;
                    det.y = (cy - h / 2 - top_pad) / scale;
                    det.width = w / scale;
                    det.height = h / scale;
                    det.confidence = max_conf;
                    det.class_id = class_id;
                    det.class_name = class_name;
                    plate_detections.push_back(det);
                } else {
                    char_boxes_for_nms.emplace_back(cx - w / 2, cy - h / 2, w, h);
                    char_confidences.push_back(max_conf);
                    char_class_ids.push_back(class_id);
                }
            }
        }

        std::vector<cv::Rect> char_boxes_for_nms_int;
        for(const auto& box : char_boxes_for_nms) {
            char_boxes_for_nms_int.emplace_back(box);
        }

        std::vector<int> char_indices;
        cv::dnn::NMSBoxes(char_boxes_for_nms_int, char_confidences, CONF_THRESHOLD, IOU_THRESHOLD, char_indices);

        std::vector<Detection> predictions_list; 
        predictions_list = plate_detections;

        if (!char_indices.empty()) {
            for (int i : char_indices) {
                cv::Rect2f box = char_boxes_for_nms[i]; 
                Detection det;
                det.x = (box.x - left_pad) / scale;
                det.y = (box.y - top_pad) / scale;
                det.width = box.width / scale;
                det.height = box.height / scale;
                det.confidence = char_confidences[i];
                det.class_id = char_class_ids[i];
                det.class_name = class_names[det.class_id];
                predictions_list.push_back(det);
            }
        }
        // --- END NMS FIX ---

        
        if (!predictions_list.empty()) {
            std::string ocr_top_row;
            for (const auto& det : predictions_list) {
                if (det.class_name == "NumberPLATE") {
                    cv::Rect plate_roi(
                        std::max(0, static_cast<int>(det.x)),
                        std::max(0, static_cast<int>(det.y)),
                        std::min(static_cast<int>(det.width), frame.cols - static_cast<int>(det.x)),
                        std::min(static_cast<int>(det.height), frame.rows - static_cast<int>(det.y))
                    );

                    if (plate_roi.width > 0 && plate_roi.height > 0) {
                        cv::Mat plate_img = frame(plate_roi);
                        
                        // --- START OCR IMPROVEMENT ---
                        cv::Rect ocr_crop_roi(
                            0, 
                            0, 
                            plate_img.cols, 
                            static_cast<int>(plate_img.rows * 0.6) // Crop to top 60%
                        );
                        cv::Mat top_half_plate = plate_img(ocr_crop_roi);
                        // --- END OCR IMPROVEMENT ---

                        cv::Mat gray_plate, thresh_plate;
                        cv::cvtColor(top_half_plate, gray_plate, cv::COLOR_BGR2GRAY);
                        cv::threshold(gray_plate, thresh_plate, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                        ocr_top_row = run_ocr(thresh_plate);
                    }
                    break; 
                }
            }

            auto plate_strings = parse_plate_detections(predictions_list, ocr_top_row);
            if (!plate_strings.first.empty() || !plate_strings.second.empty()) {
                std::string complete_plate = plate_strings.first + " " + plate_strings.second;

                // --- Producer and Cache Logic ---
                bool should_send = false;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    auto it = recent_plates.find(complete_plate);
                    if (it == recent_plates.end() || std::chrono::steady_clock::now() - it->second > CACHE_DURATION) {
                        should_send = true;
                        recent_plates[complete_plate] = std::chrono::steady_clock::now();
                    }
                }

                if (should_send) {
                    std::cout << "Queueing plate for API send: " << complete_plate << std::endl;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        plate_queue.push(complete_plate);
                    }
                    queue_cv.notify_one(); 
                }
            }
        }

    }

    // --- 6. CLEANUP ---
    std::cout << "Shutting down..." << std::endl;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        finished = true;
    }
    queue_cv.notify_one(); 
    consumer_thread.join(); 

    cap.release();
    cv::destroyAllWindows();
    curl_global_cleanup();
    std::cout << "Program finished." << std::endl;
    return 0;
}
