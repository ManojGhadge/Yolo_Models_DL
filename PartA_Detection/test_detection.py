import argparse
from ultralytics import YOLO
import cv2
import os

def main():
    parser = argparse.ArgumentParser(description="Test YOLO detection model on new data")
    parser.add_argument("image_path", help="Path to the image file to test")
    parser.add_argument("--model_path", default="D:/DEEP_LEARNING/YOLO-ASSIGNMENT/PartA_Detection/runs/detect/runs/detect/helmet_detect/weights/best.pt", help="Path to the trained model")
    parser.add_argument("--output_dir", default="test_results", help="Directory to save results")

    args = parser.parse_args()

    # Load the model
    model = YOLO(args.model_path)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Perform prediction
    results = model.predict(args.image_path, save=True, project=args.output_dir, name="predictions")

    # Print results
    for result in results:
        print(f"Image: {result.path}")
        print(f"Detections: {len(result.boxes)}")
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls]
            print(f"  {class_name}: {conf:.2f}")

    print(f"Results saved in {args.output_dir}/predictions/")

if __name__ == "__main__":
    main()