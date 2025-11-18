# AI-Powered Japanese License Plate Recognition

This project is a real-time Japanese license plate recognition system designed to run efficiently on a variety of hardware, from standard CPUs to AI-accelerated devices like the Google Coral. It utilizes a custom-trained YOLO model and is written entirely in Python.

## Development History

This project has evolved significantly to arrive at the current Python-based, AI-model-driven implementation.

The initial proof-of-concept was written in Python but relied on the `EasyOCR` library for character recognition. This method proved to be too slow and inaccurate, especially for the complex Japanese Kanji characters found on license plates.

To improve accuracy, the project shifted to a custom-trained AI model based on the YOLO architecture. The first models were trained and tested, leading to a significant improvement in character and number recognition.

An early version of the application was converted to C++ to maximize performance on a Raspberry Pi 4. However, to improve maintainability, simplify the development process, and leverage modern Python AI libraries, the project has since reverted to a pure Python implementation. The current version uses the `ultralytics` library, which provides high performance and automatic hardware acceleration detection.

## System Architecture

*   **Programming Language:** Python
*   **AI Model:** Custom-trained YOLO model.
*   **Core Library:** `ultralytics` for model loading and inference.
*   **Camera Handling:** `picamera2` for native, high-performance camera access on Raspberry Pi.
*   **Hardware Acceleration:** Optional support for Google Coral Edge TPU via the `pycoral` library.

## Core Python Packages

*   **`cv2` (OpenCV):** Used for essential image processing tasks, such as converting the camera's RGB output to the BGR format expected by the model.
*   **`picamera2`:** The official library for controlling the Raspberry Pi Camera Module, providing a high-performance, low-overhead interface for video capture.
*   **`numpy`:** A fundamental package for numerical operations, used extensively for handling image data as arrays.
*   **`ultralytics`:** The core framework that loads and runs the YOLO object detection model, handling inference and hardware acceleration.
*   **`PyYAML`:** Used to load the application's settings and model configurations from the `config.yaml` file, allowing for easy adjustments without changing the code.
*   **`threading` and `queue`:** These built-in libraries are used together to run the AI model inference in a separate thread, preventing the camera feed from stuttering while waiting for a detection result.

## Technical Rationale

### Why Python instead of C++?

While C++ offers raw performance, modern Python AI libraries like `ultralytics` have closed the performance gap significantly. By using Python, the project benefits from:
*   **Rapid Development:** Python's concise syntax and vast ecosystem allow for faster prototyping and iteration.
*   **Maintainability:** Python code is generally easier to read, debug, and maintain than low-level C++ code.
*   **Automatic Hardware Acceleration:** The `ultralytics` library can automatically detect and use available hardware accelerators (like Google's Edge TPU) with no code changes, providing performance comparable to a compiled language.

### Why a Custom YOLO Model instead of OCR?

Standard Optical Character Recognition (OCR) libraries like `EasyOCR` are general-purpose and often struggle with the specific fonts, angles, and lighting conditions of license plates. A custom-trained YOLO model offers superior accuracy because it is trained specifically to recognize the characters and layout of Japanese license plates.

### Why `picamera2` for Camera Handling?

The `picamera2` library is the official, modern Python interface for the Raspberry Pi's camera modules. It replaces the older, deprecated `picamera` library and offers several advantages over generic solutions like `cv2.VideoCapture`:

*   **Official Support and Stability:** As the officially supported library, `picamera2` is optimized for the Raspberry Pi hardware and receives ongoing updates, ensuring long-term stability.
*   **High Performance:** It provides direct, low-level access to the camera's hardware capabilities, minimizing overhead and delivering the best possible performance.
*   **Advanced Configuration:** It exposes a rich set of controls for fine-tuning camera settings like exposure, white balance, and focus, which is critical for achieving high detection accuracy in varying light conditions.
*   **Seamless Integration:** It integrates smoothly with the Raspberry Pi OS and other Python libraries like OpenCV, making it the most reliable choice for computer vision projects on this platform.

### Why Raspberry Pi OS 64-bit Lite instead of Ubuntu Server?

While Ubuntu Server is a capable operating system, Raspberry Pi OS 64-bit Lite was chosen for this project for several key reasons:

*   **Hardware Optimization and Stability:** As the official OS, it is finely tuned for the Raspberry Pi's hardware, ensuring maximum stability and performance out-of-the-box.
*   **Seamless Camera Integration:** Raspberry Pi OS provides native, well-documented support for camera modules through libraries like `picamera2`. This avoids the complex configuration and potential compatibility issues that can arise on other operating systems.
*   **Guaranteed Package Availability:** Essential Python packages for hardware interaction (like GPIO control and camera access) are maintained in the official repositories, guaranteeing straightforward installation.
*   **Extensive Community Support:** The vast and active Raspberry Pi community provides a wealth of tutorials, forums, and troubleshooting guides specifically tailored to the OS, making development and problem-solving much easier.
*   **Minimal Resource Footprint:** The "Lite" version is a headless, minimal installation without a desktop environment. This conserves CPU and RAM, dedicating more system resources to the AI inference task.

### Why Overclock the CPU to 1.9 GHz?

The Raspberry Pi 4's default CPU clock speed is 1.5 GHz. Overclocking it to 1.9 GHz provides a significant performance boost of over 25%. This is crucial for real-time object detection, as it reduces the inference time and allows for a higher frame rate. The CPU is the primary processor for running the TFLite model, so a faster clock speed directly translates to faster detection.

**How to Overclock:**
*Ensure you have adequate cooling (e.g., a fan or heatsink) before proceeding.*
1.  Edit the `/boot/firmware/config.txt` file:
    ```bash
    sudo nano /boot/firmware/config.txt
    ```
2.  Add the following lines to the end of the file:
    ```
    # Overclock
    over_voltage=4
    arm_freq=1900
    ```
3.  Save the file and reboot.

## How the Program Works

The application uses a multi-threaded design to ensure a smooth, real-time video feed while processing frames for license plate detection.

1.  **Initialization:**
    *   The script loads a YOLO model (`.pt` for CPU or `.tflite` for Edge TPU).

2.  **Worker Thread:**
    *   A separate "worker" thread is launched to handle the AI inference.
    *   The main thread captures frames from the camera and places them into a `frame_queue`.
    *   The worker thread retrieves frames from the queue, runs the YOLO model prediction, and places the results (bounding boxes and class names) into a `result_queue`.
    *   This design prevents the main thread from freezing while waiting for the AI model to process a frame.

3.  **Main Thread:**
    *   The main thread continuously captures video from the camera using the `picamera2` library.
    *   It checks the `result_queue` for any new detection results from the worker thread.
    *   When a result is available, it parses the detected characters, formats them into a license plate string, and prints the result to the console.
    *   It keeps track of the last detected plate to avoid spamming the console with duplicate readings.

4.  **Parsing and Translation:**
    *   The `get_yolo_parsed_strings` function organizes the raw detections. It identifies the main "NumberPLATE" bounding box to determine the vertical position of the top and bottom rows.
    *   Characters are sorted by their horizontal position to form the correct sequence.
    *   The `detect-v2.py` script uses the `location_dictionary` to translate the location name from English to Japanese.
    *   Both `detect.py` and `detect-v2.py` now load all settings from a central `config.yaml` file. This includes model paths, camera settings, and feature flags like Hiragana detection.

## Model Training

The YOLO model was trained in Google Colab using the [Number Plate in Japan dataset from Roboflow](https://universe.roboflow.com/moriken/number-plate-in-japan).

### Training Steps in Google Colab

1.  **Set up the Environment:** Open a new Google Colab notebook and set the runtime to use a GPU accelerator (`Runtime` -> `Change runtime type` -> `T4`).
2.  **Install Dependencies:** Install the `ultralytics` library.
    ```python
    !pip install ultralytics
    ```
3.  **Train the Model:**
    ```python
    !yolo task=detect mode=train model=yolov8n.pt data=./data.yaml epochs=100 imgsz=640
    ```
4.  **Export to TFLite (for CPU):**
    ```python
    !yolo export model=runs/detect/train/weights/best.pt format=tflite
    ```
5.  **Export to Edge TPU TFLite (for Coral):**
    ```bash
    # You will need to follow Google's instructions to compile the TFLite model for the Edge TPU.
    # This step is typically done on a Linux machine with the Edge TPU Compiler installed.
    ```

## Utilities

### Camera Focus (`test/focus_camera.py`)

Before running the main detection script, it is important to ensure your camera is properly focused. A blurry image will significantly decrease detection accuracy. This script provides a simple way to view the live camera feed so you can manually adjust the lens.

**How to Use:**
1.  Navigate to the `test` directory.
2.  Run the script:
    ```bash
    python focus_camera.py
    ```
3.  A window will appear showing the camera feed. Physically adjust your camera's lens until the image is sharp and clear.
4.  Press 'q' to close the feed.

## Installation

Follow these steps to set up the project on a Raspberry Pi with Raspberry Pi OS.

### 1. System Update and OS Dependencies

First, update your system and install essential packages. `git` is needed to clone the repository, and `python3-picamera2` is the official, optimized way to install the camera library on Raspberry Pi OS.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git python3-picamera2 -y
```

### 2. Enable the Camera

The camera interface must be enabled in the Raspberry Pi configuration.

1.  Run the configuration tool:
    ```bash
    sudo raspi-config
    ```
2.  Navigate to **Interface Options** -> **Legacy Camera** and select **Yes** to enable it.
3.  Reboot when prompted.

### 3. Clone the Repository

Contact your administrator to receive the `{API-TOKEN}` required to clone the repository.

```bash
git clone https://x-bitbucket-api-token-auth:{API-TOKEN}@bitbucket.org/lixilg/ai-vision.git
cd ai-vision
```

### 4. Install Python Dependencies

Install the required Python packages using `pip`. The dependencies vary based on whether you are using a Google Coral accelerator.

#### Standard Raspberry Pi (CPU-Only)
```bash
pip install ultralytics opencv-python pyyaml
```

#### Raspberry Pi with Google Coral
```bash
pip install ultralytics "opencv-python<4.9" "tflite-runtime>=2.14" pycoral pyyaml
```

### 5. Configure the Application (`config.yaml`)

All settings for the application are managed in the `config.yaml` file. Before running, you may need to adjust the following:

*   `model_path`: Path to the YOLO model file (`.pt`).
*   `edge_tpu_model_path`: Path to the YOLO edge tpu model file (`.tflite.pt`).
*   `data_path`: Path to the `data.yaml` file from the training dataset.
*   `use_hiragana`: Set to `true` to enable detection and mapping of Hiragana characters.
*   `location_dictionary`: Maps detected location names to their Japanese Kanji representations.

## How to Run

Once the installation is complete, you can run the detection scripts.

#### Standard CPU Detection (`detect.py`)

This script runs the YOLO model on the Raspberry Pi's CPU.

**To run:**
```bash
python detect.py
```

#### Accelerated Detection with Google Coral (`detect-v2.py`)

This script is optimized for use with a [Google Coral USB Accelerator](https://coral.ai/products/accelerator/).

**To run:**
1.  Ensure your Google Coral is connected and the [Edge TPU runtime is installed](https://coral.ai/docs/accelerator/get-started/).
2.  Run the script:
```bash
python detect-v2.py
```

## Areas for Improvement

*   **API Integration:** The current script prints to the console. The multi-threaded design could easily be extended to send detected plates to a web API in a non-blocking manner.
*   **Model Pruning:** While the Edge TPU provides acceleration, further model optimization techniques like pruning could reduce the model size and potentially increase speed even more.
*   **ZRAM Integration:** For devices with limited memory, like a Raspberry Pi, using ZRAM can improve stability.
    *   **Simple Explanation:** Think of ZRAM as a way to cleverly make your device's memory bigger than it actually is. It compresses less-used data to free up space, which helps prevent the program from crashing if it needs more memory than is physically available.
    *   **Technical Explanation:** ZRAM creates a compressed block device in RAM that acts as a swap disk. This can prevent the system from running out of memory when under heavy load, improving stability without the performance cost of a traditional swap file on an SD card.

    **How to Set Up ZRAM on Raspberry Pi OS:**

    1.  **Install the ZRAM service:**
        ```bash
        sudo apt update
        sudo apt install zram-tools
        ```

    2.  **Configure ZRAM (Optional but Recommended):**
        By default, `zram-tools` allocates 50% of your RAM to the ZRAM swap. You can adjust this by editing the configuration file:
        ```bash
        sudo nano /etc/default/zramswap
        ```
        Look for the `ALGO` and `PERCENT` settings. For example, to use the `lz4` algorithm (which is faster than the default `lzo`) and allocate 75% of RAM, you would change it to:
        ```
        ALGO=lz4
        PERCENT=75
        ```

    3.  **Reboot the System:**
        For the changes to take effect, reboot your Raspberry Pi.
        ```bash
        sudo reboot
        ```

    4.  **Verify ZRAM is Active:**
        After rebooting, you can check the status of the ZRAM swap by running:
        ```bash
        swapon --show
        ```
        You should see an entry for `/dev/zram0`.
