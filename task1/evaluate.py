import argparse
import json
import os

from src.evaluator import FashionEvaluator

FILE_EVAL_CONFIG = os.path.join("config", "eval_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fashion classification model")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate on the entire training dataset instead of only the validation split",
    )
    args = parser.parse_args()

    with open(FILE_EVAL_CONFIG) as f:
        params = json.load(f)

    evaluator = FashionEvaluator(**params)
    metrics = evaluator.evaluate(evaluate_all=args.all)
    print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
