from ultralytics import YOLO
import torch

print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

model = YOLO("yolov8n.pt")   # nano = fastest on CPU

model.train(
    data="dataset/data.yaml",
    epochs=30,          # keep low for CPU
    imgsz=416,          # smaller image = faster CPU training
    batch=4,            # small batch for CPU RAM
    device="cpu",
    workers=2,
    project="runs/detect",
    name="helmet_detect",
    exist_ok=True
)
print("Done! Model saved in runs/detect/helmet_detect/weights/best.pt")