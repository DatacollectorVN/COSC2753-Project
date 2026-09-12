import argparse
import json
import os

from src.evaluator import FashionEvaluator

FILE_PREDICT_CONFIG = os.path.join("config", "predict_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion gender & usage inference (Task 3)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data",
        type=str,
        help="Path to folder of images. Predicts all images and saves predictions.csv",
    )
    group.add_argument(
        "--image",
        type=str,
        help="Path to a single image. Prints prediction to console",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV with metadata (articleType, masterCategory, baseColour) for test images",
    )
    args = parser.parse_args()

    with open(FILE_PREDICT_CONFIG) as f:
        params = json.load(f)

    if args.data:
        params["TEST_IMAGE_DIR"] = args.data
        if args.csv:
            params["TEST_CSV_PATH"] = args.csv
        evaluator = FashionEvaluator(**params)
        evaluator.predict()

    elif args.image:
        evaluator = FashionEvaluator(**params)
        result = evaluator.predict_single(args.image, csv_path=args.csv)
        print(f"Image: {os.path.basename(args.image)}")
        print(f"Gender: {result['gender']} (confidence: {result['gender_confidence']:.4f})")
        print(f"Usage:  {result['usage']} (confidence: {result['usage_confidence']:.4f})")
