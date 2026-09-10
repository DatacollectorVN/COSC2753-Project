import json
import os

from src.evaluator import FashionEvaluator


FILE_PREDICT_CONFIG = os.path.join("config", "predict_config.json")


if __name__ == "__main__":
    with open(FILE_PREDICT_CONFIG, encoding="utf-8") as config_file:
        parameters = json.load(config_file)

    evaluator = FashionEvaluator(**parameters)
    result = evaluator.predict()
    print(f"Saved {result['prediction_rows']} predictions to {result['prediction_path']}")
