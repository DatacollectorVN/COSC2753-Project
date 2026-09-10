import json
import os

from src.trainer import FashionTrainer


FILE_TRAIN_CONFIG = os.path.join("config", "train_config.json")


if __name__ == "__main__":
    with open(FILE_TRAIN_CONFIG, encoding="utf-8") as config_file:
        parameters = json.load(config_file)

    trainer = FashionTrainer(**parameters)
    summary = trainer.train()
    print(f"Best CV macro-F1: {summary['best_cv_macro_f1']:.4f}")
    print(f"Holdout macro-F1: {summary['holdout_macro_f1']:.4f}")
    print(f"Final model: {summary['final_model_path']}")
