import json
import os

from src.trainer import FashionTrainer

FILE_TRAIN_CONFIG = os.path.join("config", "train_config.json")

if __name__ == "__main__":
    with open(FILE_TRAIN_CONFIG) as f:
        params = json.load(f)

    trainer = FashionTrainer(**params)
    trainer.train()
