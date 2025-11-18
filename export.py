from ultralytics import YOLO

# Load your PyTorch model
model = YOLO('ai-model/best.pt') 

# Export to Edge TPU format
# Specify an image size (imgsz) that matches your training, e.g., 640
model.export(format='edgetpu', imgsz=640)