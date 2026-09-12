import argparse
import json
import os

from src.evaluator import FashionEvaluator

FILE_PREDICT_CONFIG = os.path.join("config", "predict_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion classification inference")
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
    args = parser.parse_args()

    with open(FILE_PREDICT_CONFIG) as f:
        params = json.load(f)

    if args.data:
        params["TEST_IMAGE_DIR"] = args.data
        evaluator = FashionEvaluator(**params)
        evaluator.predict()

    elif args.image:
        evaluator = FashionEvaluator(**params)
        result = evaluator.predict_single(args.image)
        print(f"Image: {os.path.basename(args.image)}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
