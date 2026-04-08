import argparse
from ultralytics import YOLO
import cv2
import os

def main():
    parser = argparse.ArgumentParser(description="Test YOLO classification model on new data")
    parser.add_argument("image_path", help="Path to the image file to test")
    parser.add_argument("--model_path", default="D:\\DEEP_LEARNING\\YOLO-ASSIGNMENT\\PartB_Classification\\runs\\classify\\runs\\classify\\fruit_classify\\weights\\best.pt", help="Path to the trained model")
    parser.add_argument("--output_dir", default="test_results", help="Directory to save results")
#D:\DEEP_LEARNING\YOLO-ASSIGNMENT\PartB_Classification\runs\classify\runs\classify\fruit_classify\weights
#runs/classify/fruit_classify/weights/best.pt
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
        if hasattr(result, 'probs') and result.probs is not None:
            # For classification, get top probabilities
            probs = result.probs
            top5 = probs.top5
            top5conf = probs.top5conf
            for i, (cls, conf) in enumerate(zip(top5, top5conf)):
                class_name = model.names[cls]
                print(f"  {i+1}. {class_name}: {conf:.2f}")
        else:
            print("  No classification results available")

    print(f"Results saved in {args.output_dir}/predictions/")

if __name__ == "__main__":
    main()