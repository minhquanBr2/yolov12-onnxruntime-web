import json
from ultralytics import YOLO

# 1. Load your specific model
model = YOLO("yolo_vehicle_detection_model.pt")  # or your custom model path

# 2. Extract properties from the model object
# model.names is a dictionary {0: 'person', 1: 'bicycle', ...}
class_list = list(model.names.values())

# 3. Build your custom metadata dictionary
metadata = {
    "inputSize": [640, 640],
    "classes": class_list,
    "confidenceThreshold": 0.2,  # Default for YOLOv8
    "nmsThreshold": 0.4,
    "modelName": f"YOLOv8 Nano (Custom)",
    "version": "8.0.0", 
    "description": "Custom YOLOv8 model optimized for mobile/edge devices.",
    "author": "Ultralytics / Your Name",
    "license": "AGPL-3.0",
    "framework": "PyTorch",
    "parameters": {
        "numParameters (M)": sum(p.numel() for p in model.model.parameters()) / 1e6
    }
}

# 4. Save to JSON
with open("model-metadata-vehicle-detection.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Metadata file generated: model-metadata-vehicle-detection.json")