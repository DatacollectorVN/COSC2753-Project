import json
import os

from src.evaluator import FashionEvaluator

FILE_PREDICT_CONFIG = os.path.join("config", "predict_config.json")

if __name__ == "__main__":
    with open(FILE_PREDICT_CONFIG) as f:
        params = json.load(f)

    evaluator = FashionEvaluator(**params)
    evaluator.predict()
