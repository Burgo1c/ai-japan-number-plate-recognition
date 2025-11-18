import time
from picamera2 import Picamera2, Preview
from libcamera import controls

# --- 1. INITIALIZE CAMERA ---
print("Initializing camera...")
picam2 = Picamera2()

# --- 2. CONFIGURE PREVIEW ---
# Create a configuration for the preview window
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)

# Start the preview using Qt for the backend
picam2.start_preview(Preview.QTGL)

print("Preview started. Setting autofocus...")

# --- 3. SET AUTOFOCUS ---
# Set the autofocus mode to continuous
# This will make the camera continuously adjust focus
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Start the camera stream
picam2.start()

print("Camera running with continuous autofocus.")
print("The preview window will remain open. Press Ctrl+C in the terminal to exit.")

# --- 4. MAIN LOOP ---
# Keep the script running to display the preview
try:
    while True:
        # The camera preview is handled by a separate thread,
        # so we just need to keep the main script alive.
        time.sleep(1)
except KeyboardInterrupt:
    # Handle Ctrl+C gracefully
    print("\nCaught Ctrl+C. Shutting down...")
finally:
    # --- 5. CLEANUP ---
    print("Stopping camera and closing preview.")
    picam2.stop_preview()
    picam2.stop()
    print("Script finished.")
