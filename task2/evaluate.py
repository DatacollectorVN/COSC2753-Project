import json
import os

from src.evaluator import FashionEvaluator


FILE_EVAL_CONFIG = os.path.join("config", "eval_config.json")


if __name__ == "__main__":
    with open(FILE_EVAL_CONFIG, encoding="utf-8") as config_file:
        parameters = json.load(config_file)

    evaluator = FashionEvaluator(**parameters)
    metrics = evaluator.evaluate()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
