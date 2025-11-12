cmake_minimum_required(VERSION 3.16)
project(YoloTFLiteDetector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Find OpenCV ---
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# --- 2. Find YAML-CPP ---
find_package(yaml-cpp REQUIRED)
message(STATUS "YAML-CPP include dir: ${yaml-cpp_INCLUDE_DIR}")
message(STATUS "YAML-CPP library: ${yaml-cpp_LIBRARY}")

# --- 3. Find TFLite (Installed by the .deb package) ---
find_path(TFLITE_INCLUDE_DIR
    NAMES "tensorflow/lite/interpreter.h"
    PATHS "/usr/include" # Standard path where the .deb installs headers
)
find_library(TFLITE_LIB
    NAMES "tensorflow-lite"
    PATHS "/usr/lib" "/usr/lib/aarch64-linux-gnu" # Standard library paths
)

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIB)
    message(FATAL_ERROR "TensorFlow Lite library not found. Did you run 'sudo apt install ./tensorflow-lite_64.deb'?")
endif()

message(STATUS "TFLite include dir: ${TFLITE_INCLUDE_DIR}")
message(STATUS "TFLite library: ${TFLITE_LIB}")

include_directories(${TFLITE_INCLUDE_DIR})

# --- 4. Find cURL (for API calls) ---
find_package(CURL REQUIRED)
message(STATUS "CURL library: ${CURL_LIBRARIES}")
message(STATUS "CURL include: ${CURL_INCLUDE_DIRS}")
include_directories(${CURL_INCLUDE_DIRS})


# --- 5. Add Executable ---
add_executable(detector main.cpp)

# --- 6. Define Project Source Directory ---
# This passes the top-level source directory to the C++ code
# so it can build absolute paths to resources.
target_compile_definitions(detector PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# --- 7. Link Libraries ---
target_link_libraries(detector
    PRIVATE
    ${OpenCV_LIBS}
    ${TFLITE_LIB}
    yaml-cpp::yaml-cpp
    CURL::libcurl
    pthread # Needed for TFLite threads and std::thread
    dl      # Needed by TFLite
)
