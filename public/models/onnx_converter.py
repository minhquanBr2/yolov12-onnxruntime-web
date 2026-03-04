from ultralytics import YOLO

# Load a model
model = YOLO("yolo_vehicle_detection_model.pt")  # load an official model

# Export the model
model.export(
    format="onnx",
    imgsz=640,           # Standard for mobile performance
    simplify=True,       # Essential for NPU compatibility
    opset=12,            # Best balance of features and mobile support
)