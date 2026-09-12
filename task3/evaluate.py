import argparse
import json
import os

from src.evaluator import FashionEvaluator

FILE_EVAL_CONFIG = os.path.join("config", "eval_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fashion gender & usage model")
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
    print(f"Gender Accuracy:   {metrics['gender']['overall_accuracy']:.4f}")
    print(f"Occasion Accuracy: {metrics['occasion']['overall_accuracy']:.4f}")
