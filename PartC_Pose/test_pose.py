import argparse
from ultralytics import YOLO
import cv2
import os

def main():
    parser = argparse.ArgumentParser(description="Test YOLO pose detection model on new data")
    parser.add_argument("image_path", help="Path to the image file to test")
    parser.add_argument("--model_path", default="D:\\DEEP_LEARNING\\YOLO-ASSIGNMENT\\PartC_Pose\\runs\\pose\\runs\\pose\\human_pose\\weights\\best.pt", help="Path to the trained model")
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
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints
            print(f"Detected {len(keypoints)} pose(s)")
            for i, kpt in enumerate(keypoints):
                print(f"  Pose {i+1}: {len(kpt)} keypoints")
                # You can access individual keypoints here if needed
        else:
            print("  No pose keypoints detected")

    print(f"Results saved in {args.output_dir}/predictions/")

if __name__ == "__main__":
    main()