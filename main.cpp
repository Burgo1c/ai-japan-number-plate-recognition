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

// Helper macro to convert preprocessor definition to string
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// --- 1. CONFIGURATION ---
const std::string MODEL_PATH = "/home/raspi/ai-japan-number-plate-recognition/ai-model/best_int8.tflite";
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

// --- Performance Implementation: Globals for Queue and Cache ---
std::queue<std::string> plate_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
bool finished = false;

std::map<std::string, std::chrono::steady_clock::time_point> recent_plates;
const std::chrono::seconds CACHE_DURATION(30); // Don't resend the same plate for 30 seconds

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
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); // 10 second timeout

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
// This function runs in a separate thread, consuming plates from the queue.
void api_consumer_thread() {
    while (true) {
        std::string plate_to_send;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // Wait until the queue has something or the program is finished
            queue_cv.wait(lock, []{ return !plate_queue.empty() || finished; });

            // If the program is finished and the queue is empty, exit the thread
            if (finished && plate_queue.empty()) {
                break;
            }

            plate_to_send = plate_queue.front();
            plate_queue.pop();
        } // Lock is released here

        //send_to_api(plate_to_send);
    }
    std::cout << "API consumer thread finished." << std::endl;
}


// --- 2. PARSING FUNCTION (MODIFIED) ---
std::string parse_plate_detections(const std::vector<Detection>& predictions) {
    float divider_y = -1.0f;

    // Find the license plate and calculate its vertical center.
    for (const auto& p : predictions) {
        if (p.class_name == "NumberPLATE") {
            divider_y = p.y + p.height / 2.0f;
            break;
        }
    }

    // If no plate is found, we can't proceed.
    if (divider_y == -1.0f) {
        return ""; // Return an empty string
    }

    // We only need to store the bottom row
    std::vector<Detection> bottom_row_detections;
    for (const auto& p : predictions) {
        if (p.class_name == "NumberPLATE") continue;
        
        // Use the character's vertical center for comparison.
        // If it's not less than the divider, it's in the bottom row.
        if ((p.y + p.height / 2.0f) >= divider_y) {
            bottom_row_detections.push_back(p);
        }
    }

    // Sort only the bottom row
    std::sort(bottom_row_detections.begin(), bottom_row_detections.end(), [](const Detection& a, const Detection& b) { return a.x < b.x; });

    std::stringstream bottom_ss;
    for (const auto& p : bottom_row_detections) {
        auto it = location_dictionary.find(p.class_name);
        bottom_ss << (it != location_dictionary.end() ? it->second : p.class_name);
    }

    std::string bottom_string = bottom_ss.str();

    std::cout << "\n--- Parsed License Plate ---" << std::endl;
    std::cout << "Bottom Row: " << bottom_string << std::endl; // Changed to only show bottom row

    return bottom_string; // Return only the bottom string
}


int main() {
    // Register signal handler for Ctrl+C (SIGINT)
    signal(SIGINT, signal_handler);

    curl_global_init(CURL_GLOBAL_ALL);

    // --- Start the consumer thread ---
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

    // On a 4-core device like a Raspberry Pi 4, using 3 threads for inference
    // leaves one core free for the OS, the main loop, and the API consumer thread.
    // This prevents CPU contention and improves overall stability.
    interpreter->SetNumThreads(3);
    interpreter->AllocateTensors();

    int input_tensor_idx = interpreter->inputs()[0];
    bool is_int8_model = (interpreter->tensor(input_tensor_idx)->type == kTfLiteUInt8);
    std::cout << "Model loaded." << std::endl;

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

        if (is_int8_model) {
            memcpy(interpreter->typed_tensor<uint8_t>(input_tensor_idx), input_img.data, input_img.total() * input_img.elemSize());
        } else {
            cv::Mat float_img;
            input_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
            memcpy(interpreter->typed_tensor<float>(input_tensor_idx), float_img.data, float_img.total() * float_img.elemSize());
        }

        interpreter->Invoke();

        const float* output_data = interpreter->typed_output_tensor<float>(0);
        TfLiteIntArray* output_dims = interpreter->tensor(interpreter->outputs()[0])->dims;
        const int num_proposals = output_dims->data[2];
        const int num_classes = output_dims->data[1] - 4;

        std::vector<cv::Rect2f> boxes_for_nms;
        std::vector<float> confidences;
        std::vector<int> class_ids;

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
                float cx = output_data[i + 0 * num_proposals];
                float cy = output_data[i + 1 * num_proposals];
                float w = output_data[i + 2 * num_proposals];
                float h = output_data[i + 3 * num_proposals];
                boxes_for_nms.emplace_back(cx - w / 2, cy - h / 2, w, h);
                confidences.push_back(max_conf);
                class_ids.push_back(class_id);
            }
        }

        // Convert Rect2f to Rect for NMSBoxes compatibility with older OpenCV versions
        std::vector<cv::Rect> boxes_for_nms_int;
        for(const auto& box : boxes_for_nms) {
            boxes_for_nms_int.emplace_back(box);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes_for_nms_int, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);

        std::vector<Detection> predictions_list;
        if (!indices.empty()) {
            for (int i : indices) {
                cv::Rect2f box = boxes_for_nms[i];
                Detection det;
                det.x = (box.x - left_pad) / scale;
                det.y = (box.y - top_pad) / scale;
                det.width = box.width / scale;
                det.height = box.height / scale;
                det.confidence = confidences[i];
                det.class_id = class_ids[i];
                det.class_name = class_names[det.class_id];
                predictions_list.push_back(det);
            }
        }

        // <--- ADD THIS DEBUG CODE --->
        std::cout << "--- RAW DETECTIONS (" << predictions_list.size() << ") ---" << std::endl;
        for (const auto& det : predictions_list) {
            std::cout << "  Class: " << det.class_name
                      << ", Conf: " << det.confidence << std::endl;
        }
        // <--- END DEBUG CODE --->
        
        // --- THIS SECTION IS MODIFIED ---
        if (!predictions_list.empty()) {
            // Call the modified function which now returns a single string
            std::string bottom_row_plate = parse_plate_detections(predictions_list);
            
            // Check if the returned bottom row string is not empty
            if (!bottom_row_plate.empty()) {
                // The "complete_plate" is now just the bottom row
                std::string complete_plate = bottom_row_plate; 

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
                    queue_cv.notify_one(); // Notify the consumer thread
                }
            }
        }
        // --- END OF MODIFIED SECTION ---

    }

    // --- 6. CLEANUP ---
    std::cout << "Shutting down..." << std::endl;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        finished = true;
    }
    queue_cv.notify_one(); // Wake up consumer thread to exit
    consumer_thread.join(); // Wait for the consumer thread to finish

    cap.release();
    cv::destroyAllWindows();
    curl_global_cleanup();
    std::cout << "Program finished." << std::endl;
    return 0;
}
