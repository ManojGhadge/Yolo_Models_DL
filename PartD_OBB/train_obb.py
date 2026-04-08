from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")

model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=416,
    batch=4,
    device="cpu",
    workers=2,
    project="runs/obb",
    name="aerial_obb",
    exist_ok=True
)
print("Done! Model saved in runs/obb/aerial_obb/weights/best.pt")