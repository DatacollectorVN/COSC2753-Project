import json
import os

from src.evaluator import FashionEvaluator

FILE_EVAL_CONFIG = os.path.join("config", "eval_config.json")

if __name__ == "__main__":
    with open(FILE_EVAL_CONFIG) as f:
        params = json.load(f)

    evaluator = FashionEvaluator(**params)
    metrics = evaluator.evaluate()
    print(f"Accuracy: {metrics['report']['accuracy']:.4f}")
