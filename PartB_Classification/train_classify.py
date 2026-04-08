from ultralytics import YOLO
import torch

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset",     # folder path — no yaml needed for classification
    epochs=20,
    imgsz=224,
    batch=8,
    device="cpu",
    workers=2,
    project="runs/classify",
    name="fruit_classify",
    exist_ok=True
)
print("Done! Model saved in runs/classify/fruit_classify/weights/best.pt")