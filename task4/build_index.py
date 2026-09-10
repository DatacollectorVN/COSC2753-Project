"""Build the embedding index for all training images.

Usage:
    python build_index.py                              # uses default config
    python build_index.py --config config/build_config.json

One-time process:
    1. Load all training images from DATA_DIR/IMAGE_SUBDIR.
    2. Extract CLIP embeddings in batches.
    3. Save embeddings + image IDs to INDEX_DIR.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from src.feature_extractor import CLIPFeatureExtractor
from src.embedding_store import EmbeddingStore
from src.utils import SettingConfig, setup_logger, set_seed


def main():
    parser = argparse.ArgumentParser(description="Build visual search index")
    parser.add_argument("--config", type=str, default="config/build_config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = SettingConfig(**json.load(f))

    set_seed()

    index_dir = Path(cfg.INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("build_index", index_dir / "build_log.txt")

    # ------------------------------------------------------------------ #
    #  Collect image paths
    # ------------------------------------------------------------------ #
    data_dir = Path(cfg.DATA_DIR)
    image_dir = data_dir / cfg.IMAGE_SUBDIR
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    logger.info(f"Found {len(image_paths)} images in {image_dir}")

    # ------------------------------------------------------------------ #
    #  Extract embeddings
    # ------------------------------------------------------------------ #
    logger.info(f"Loading model: {cfg.MODEL_NAME}")
    extractor = CLIPFeatureExtractor(model_name=cfg.MODEL_NAME)
    logger.info(f"Embedding dim: {extractor.embedding_dim}, device: {extractor.device}")

    store = EmbeddingStore(embedding_dim=extractor.embedding_dim)
    batch_size = cfg.BATCH_SIZE

    t_start = time.time()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        images = []
        valid_ids = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_ids.append(p.stem)
            except Exception as e:
                logger.warning(f"Skipping {p.name}: {e}")

        if not images:
            continue

        embeddings = extractor.extract_batch(images)
        store.add_batch(valid_ids, embeddings)

        processed = min(i + batch_size, len(image_paths))
        if processed % (batch_size * 10) == 0 or processed == len(image_paths):
            elapsed = time.time() - t_start
            rate = processed / elapsed
            logger.info(f"  [{processed}/{len(image_paths)}] {rate:.0f} img/s")

    elapsed = time.time() - t_start
    logger.info(f"Embedded {store.size} images in {elapsed:.1f}s ({store.size / elapsed:.0f} img/s)")

    # ------------------------------------------------------------------ #
    #  Save index
    # ------------------------------------------------------------------ #
    store.save(index_dir)
    logger.info(f"Index saved to {index_dir}")

    # Save build metadata
    meta = {
        "model_name": cfg.MODEL_NAME,
        "embedding_dim": extractor.embedding_dim,
        "num_images": store.size,
        "build_time_s": round(elapsed, 1),
    }
    with open(index_dir / "build_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
