#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <yaml-cpp/yaml.h>

// --- 1. CONFIGURATION ---
const std::string MODEL_PATH = "best_int8.tflite";
const std::string DATA_YAML_PATH = "data.yaml";
const std::string IMAGE_PATH = "plate1.jpg";

// Model parameters
const int IMG_SIZE = 640;
const float CONF_THRESHOLD = 0.5f;
const float IOU_THRESHOLD = 0.45f;

// --- Location Dictionary (from config.py) ---
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

// --- 2. PARSING FUNCTION ---
void parse_plate_detections(const std::vector<Detection>& predictions) {
    float divider_y = -1.0f;

    // Find the 'NumberPLATE' detection
    for (const auto& p : predictions) {
        if (p.class_name == "NumberPLATE") {
            divider_y = p.y;
            break;
        }
    }

    if (divider_y == -1.0f) {
        std::cerr << "Error: 'NumberPLATE' detection not found." << std::endl;
        return;
    }

    std::vector<Detection> top_row, bottom_row;
    for (const auto& p : predictions) {
        if (p.class_name == "NumberPLATE") continue;
        if (p.y < divider_y) {
            top_row.push_back(p);
        } else {
            bottom_row.push_back(p);
        }
    }

    // Sort rows by x-coordinate
    std::sort(top_row.begin(), top_row.end(), [](const Detection& a, const Detection& b) {
        return a.x < b.x;
    });
    std::sort(bottom_row.begin(), bottom_row.end(), [](const Detection& a, const Detection& b) {
        return a.x < b.x;
    });

    // Build strings
    std::string top_string, bottom_string;
    for (const auto& p : top_row) {
        auto it = location_dictionary.find(p.class_name);
        top_string += (it != location_dictionary.end() ? it->second : p.class_name) + " ";
    }
    for (const auto& p : bottom_row) {
        auto it = location_dictionary.find(p.class_name);
        bottom_string += (it != location_dictionary.end() ? it->second : p.class_name) + " ";
    }

    std::cout << "\n--- Parsed License Plate ---" << std::endl;
    std::cout << "Top Row:    " << top_string << std::endl;
    std::cout << "Bottom Row: " << bottom_string << std::endl << std::endl;
}


int main() {
    // --- 3. INITIALIZE MODEL & LOAD DATA ---
    std::cout << "Loading TFLite model..." << std::endl;

    // Load class names
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

    // Load TFLite model
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

    interpreter->SetNumThreads(4);

    interpreter->AllocateTensors();

    // Get input and output details
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteIntArray* input_dims = interpreter->tensor(input_tensor_idx)->dims;
    bool is_int8_model = (interpreter->tensor(input_tensor_idx)->type == kTfLiteUInt8);

    std::cout << "Model loaded." << std::endl;

    // --- 4. LOAD, PRE-PROCESS, AND RUN ---
    std::cout << "Loading image: " << IMAGE_PATH << std::endl;
    cv::Mat frame = cv::imread(IMAGE_PATH);
    if (frame.empty()) {
        std::cerr << "Error: Could not read image." << std::endl;
        return -1;
    }

    std::cout << "Running detection..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Pre-processing
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    float scale = std::min(static_cast<float>(IMG_SIZE) / orig_h, static_cast<float>(IMG_SIZE) / orig_w);
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);

    cv::Mat resized_img;
    cv::resize(frame, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat input_img(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    int top_pad = (IMG_SIZE - new_h) / 2;
    int left_pad = (IMG_SIZE - new_w) / 2;
    resized_img.copyTo(input_img(cv::Rect(left_pad, top_pad, new_w, new_h)));

    // Prepare input tensor
    if (is_int8_model) {
        uint8_t* input_tensor = interpreter->typed_tensor<uint8_t>(input_tensor_idx);
        memcpy(input_tensor, input_img.data, input_img.total() * input_img.elemSize());
    } else {
        cv::Mat float_img;
        input_img.convertTo(float_img, CV_32F, 1.0 / 255.0);
        float* input_tensor = interpreter->typed_tensor<float>(input_tensor_idx);
        memcpy(input_tensor, float_img.data, float_img.total() * float_img.elemSize());
    }

    // Run Inference
    interpreter->Invoke();

    // Post-processing
    TfLiteIntArray* output_dims = interpreter->tensor(interpreter->outputs()[0])->dims;
    const int num_proposals = output_dims->data[2]; // e.g., 8400
    const int num_elements_per_proposal = output_dims->data[1]; // e.g., 84 (num_classes + 4)
    const int num_classes = num_elements_per_proposal - 4;
    const float* output_data = interpreter->typed_output_tensor<float>(0);

    std::vector<cv::Rect2f> boxes_for_nms;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_proposals; ++i) {
        // Extract box coordinates (cx, cy, w, h) from the planar output
        float cx = output_data[i + 0 * num_proposals];
        float cy = output_data[i + 1 * num_proposals];
        float w = output_data[i + 2 * num_proposals];
        float h = output_data[i + 3 * num_proposals];

        // Find the class with the highest score
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
            // Convert [cx, cy, w, h] to [x1, y1, w, h] for NMS
            boxes_for_nms.emplace_back(cx - w / 2, cy - h / 2, w, h);
            confidences.push_back(max_conf);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_for_nms, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);

    std::vector<Detection> predictions_list;
    if (!indices.empty()) {
        for (int i : indices) {
            cv::Rect2f box = boxes_for_nms[i];

            float x1_unpad = box.x - left_pad;
            float y1_unpad = box.y - top_pad;

            float x1_orig = x1_unpad / scale;
            float y1_orig = y1_unpad / scale;
            float w_orig = box.width / scale;
            float h_orig = box.height / scale;

            float x1_clipped = std::max(0.0f, x1_orig);
            float y1_clipped = std::max(0.0f, y1_orig);
            float x2_clipped = std::min(static_cast<float>(orig_w), x1_orig + w_orig);
            float y2_clipped = std::min(static_cast<float>(orig_h), y1_orig + h_orig);

            Detection det;
            det.width = x2_clipped - x1_clipped;
            det.height = y2_clipped - y1_clipped;
            det.x = x1_clipped + det.width / 2;
            det.y = y1_clipped + det.height / 2;
            det.confidence = confidences[i];
            det.class_id = class_ids[i];
            det.class_name = class_names[det.class_id];
            predictions_list.push_back(det);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Finished detection in " << elapsed.count() << " seconds." << std::endl;

    // --- 5. PARSE RESULTS ---
    if (!predictions_list.empty()) {
        std::cout << "Found " << predictions_list.size() << " total detections." << std::endl;
        parse_plate_detections(predictions_list);
    } else {
        std::cout << "No detections found." << std::endl;
    }

    return 0;
}
