cmake_minimum_required(VERSION 3.10)
project(YOLO_Detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---

# 1. OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR.")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# 2. TensorFlow Lite
# Note: You must have built TFLite from source or downloaded the pre-built library.
# Set TENSORFLOW_LITE_DIR to the root of your TFLite installation.
set(TENSORFLOW_LITE_DIR "" CACHE PATH "Path to TensorFlow Lite installation")
if(NOT TENSORFLOW_LITE_DIR)
    message(FATAL_ERROR "TENSORFLOW_LITE_DIR is not set. Please point it to your TFLite build/install directory.")
endif()

include_directories(${TENSORFLOW_LITE_DIR}/include)
# The TFLite library is often named libtensorflow-lite.a
# The exact name might vary depending on your build.
find_library(TFLITE_LIB NAMES tensorflow-lite PATHS ${TENSORFLOW_LITE_DIR}/lib)
if(NOT TFLITE_LIB)
    message(FATAL_ERROR "TensorFlow Lite library not found in ${TENSORFLOW_LITE_DIR}/lib")
endif()

# 3. YAML-CPP
# You can install this via a package manager (e.g., vcpkg, apt, brew) or build from source.
find_package(yaml-cpp REQUIRED)
if(NOT yaml-cpp_FOUND)
    message(FATAL_ERROR "yaml-cpp not found. Please install it or set yaml-cpp_DIR.")
endif()

# --- Build Executable ---

add_executable(detector main.cpp)

# Link all libraries
target_link_libraries(detector
    PRIVATE
    ${OpenCV_LIBS}
    ${TFLITE_LIB}
    yaml-cpp
    # TFLite often requires these additional libraries
    pthread
    dl
)

# --- Installation (Optional) ---
install(TARGETS detector DESTINATION bin)
