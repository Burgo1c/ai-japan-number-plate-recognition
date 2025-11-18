import cv2

# --- 1. CONFIGURATION ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Note: On Raspberry Pi, camera index 0 might be the CSI camera or a USB camera.
# If you have multiple cameras, you might need to change this index.
CAMERA_INDEX = 0

# --- 2. INITIALIZE CAMERA ---
print("Starting camera feed...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera.")
    print("Please ensure the camera is connected and drivers are installed.")
    print(f"Tried to open camera with index: {CAMERA_INDEX}")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Camera started. Press 'q' to quit.")

# --- 3. MAIN LOOP ---
try:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame. Exiting.")
            break

        # Display the frame
        cv2.imshow('Camera Feed - Press "q" to quit', frame)

        # Wait for the 'q' key to be pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nCaught Ctrl+C. Shutting down...")

# --- 4. CLEANUP ---
finally:
    print("Releasing camera and closing windows.")
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")
