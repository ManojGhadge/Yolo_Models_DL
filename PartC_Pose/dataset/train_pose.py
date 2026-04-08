from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

model.train(
    data="dataset/data.yaml",   # or "coco8-pose.yaml" as fallback
    epochs=30,
    imgsz=416,
    batch=4,
    device="cpu",
    workers=2,
    project="runs/pose",
    name="human_pose",
    exist_ok=True
)
print("Done! Model saved in runs/pose/human_pose/weights/best.pt")