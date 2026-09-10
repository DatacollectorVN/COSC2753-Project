"""Search for visually similar fashion items given a query image.

Usage:
    python search.py --query <path_to_image>
    python search.py --query <path_to_image> --k 10 --mode ann
    python search.py --query <path_to_image> --save results/search/output.png

Pipeline:
    1. Load prebuilt embedding index from INDEX_DIR.
    2. Embed the query image with CLIP.
    3. Search for Top-K similar items (KNN or ANN).
    4. Display/save results with similarity scores.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.feature_extractor import CLIPFeatureExtractor
from src.embedding_store import EmbeddingStore
from src.utils import SettingConfig, plot_search_results


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load image as RGB numpy array for display."""
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    with open("config/search_config.json") as f:
        cfg = SettingConfig(**json.load(f))

    parser = argparse.ArgumentParser(description="Fashion visual search")
    parser.add_argument("--query", type=str, required=True, help="Path to query image")
    parser.add_argument("--k", type=int, default=cfg.TOP_K, help="Number of results to return")
    parser.add_argument("--mode", type=str, default=cfg.SEARCH_MODE, choices=["knn", "ann"], help="Search mode")
    parser.add_argument("--save", type=str, default=None, help="Path to save result visualization")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing dataset images (for display)")
    args = parser.parse_args()

    top_k = args.k
    search_mode = args.mode

    t_total_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    #  Load index
    # ------------------------------------------------------------------ #
    print(f"Loading index from {cfg.INDEX_DIR}...")
    store = EmbeddingStore()
    store.load(cfg.INDEX_DIR)
    print(f"Loaded {store.size} embeddings (dim={store.embedding_dim})")

    if search_mode == "ann":
        print("Building ANN index...")
        hnsw_m = getattr(cfg, "HNSW_M", 32)
        hnsw_ef = getattr(cfg, "HNSW_EF_SEARCH", 64)
        store.build_ann_index(m=hnsw_m, ef_search=hnsw_ef)
        print(f"Built ANN index (HNSW m={hnsw_m}, ef_search={hnsw_ef})")
    else:
        print("Building KNN index...")
        store.build_knn_index()
        print("Built KNN index (exact search)")

    # ------------------------------------------------------------------ #
    #  Embed query
    # ------------------------------------------------------------------ #
    print(f"Loading model: {cfg.MODEL_NAME}")
    extractor = CLIPFeatureExtractor(model_name=cfg.MODEL_NAME)

    query_image = Image.open(args.query).convert("RGB")
    query_embedding = extractor.extract(query_image)

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #
    t_search_start = time.perf_counter()
    results = store.search(query_embedding, k=top_k, mode=search_mode)
    t_search = (time.perf_counter() - t_search_start) * 1000  # ms

    t_total = (time.perf_counter() - t_total_start) * 1000  # ms

    print(f"\nTop-{top_k} results ({search_mode.upper()}):")
    print("-" * 40)
    for rank, (image_id, score) in enumerate(results, 1):
        print(f"  #{rank}  {image_id}  score={score:.4f}")
    print("-" * 40)
    print(f"Search time:  {t_search:.2f} ms")
    print(f"Total time:   {t_total:.2f} ms")

    # ------------------------------------------------------------------ #
    #  Visualize
    # ------------------------------------------------------------------ #
    save_path = args.save
    image_dir = args.image_dir

    if save_path and image_dir:
        image_dir = Path(image_dir)
        query_rgb = load_image_rgb(args.query)

        result_images = []
        result_scores = []
        result_ids = []
        for image_id, score in results:
            # Try common extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = image_dir / f"{image_id}{ext}"
                if candidate.exists():
                    result_images.append(load_image_rgb(candidate))
                    result_scores.append(score)
                    result_ids.append(image_id)
                    break

        plot_search_results(query_rgb, result_images, result_scores, result_ids, Path(save_path))
        print(f"\nVisualization saved to {save_path}")


if __name__ == "__main__":
    main()
