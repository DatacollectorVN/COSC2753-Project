"""Embedding store with FAISS-backed KNN and ANN search.

Stores image embeddings and their associated image IDs on disk.
Supports two search modes:
    - KNN:  Exact search via FAISS IndexFlatIP (brute-force cosine).
    - ANN:  Approximate search via FAISS IndexHNSWFlat (graph-based).

All embeddings are expected to be L2-normalized so that inner product = cosine similarity.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np


class EmbeddingStore:
    """Persistent vector store for fashion image embeddings.

    Attributes:
        embedding_dim: Dimensionality of embedding vectors.
        embeddings: np.ndarray of shape (N, embedding_dim) or None.
        image_ids: List of image identifiers corresponding to each row.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.embeddings: np.ndarray | None = None
        self.image_ids: list[str] = []

        self._knn_index: faiss.Index | None = None
        self._ann_index: faiss.Index | None = None

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #

    def add(self, image_id: str, embedding: np.ndarray) -> None:
        """Add a single embedding to the store (before building index)."""
        embedding = embedding.reshape(1, -1).astype(np.float32)
        if self.embeddings is None:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.image_ids.append(image_id)

    def add_batch(self, image_ids: list[str], embeddings: np.ndarray) -> None:
        """Add a batch of embeddings to the store."""
        embeddings = embeddings.astype(np.float32)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.image_ids.extend(image_ids)

    def build_knn_index(self) -> None:
        """Build exact search index (IndexFlatIP — inner product on L2-normed vectors = cosine)."""
        assert self.embeddings is not None, "No embeddings to index."
        self._knn_index = faiss.IndexFlatIP(self.embedding_dim)

        if faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            self._knn_index = faiss.index_cpu_to_gpu(gpu_res, 0, self._knn_index)

        self._knn_index.add(self.embeddings)

    def build_ann_index(self, m: int = 32, ef_search: int = 64) -> None:
        """Build approximate search index (HNSW graph-based).

        Args:
            m: Number of neighbors per node in the HNSW graph (higher = more accurate, slower build).
            ef_search: Search depth at query time (higher = more accurate, slower query).
        """
        assert self.embeddings is not None, "No embeddings to index."
        # Force single-thread to avoid OpenMP deadlock on macOS
        prev_threads = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        faiss.omp_set_num_threads(1)

        self._ann_index = faiss.IndexHNSWFlat(self.embedding_dim, m, faiss.METRIC_INNER_PRODUCT)
        self._ann_index.hnsw.efSearch = ef_search
        self._ann_index.add(self.embeddings)

        # Restore threads
        if prev_threads is not None:
            os.environ["OMP_NUM_THREADS"] = prev_threads
        else:
            os.environ.pop("OMP_NUM_THREADS", None)
        faiss.omp_set_num_threads(faiss.omp_get_max_threads())

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def search_knn(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Exact nearest-neighbor search via brute-force cosine similarity.

        Args:
            query: L2-normalized query vector of shape (embedding_dim,).
            k: Number of results to return.

        Returns:
            List of (image_id, similarity_score) tuples, descending by score.
        """
        if self._knn_index is None:
            self.build_knn_index()

        query = query.reshape(1, -1).astype(np.float32)
        scores, indices = self._knn_index.search(query, k)
        return [(self.image_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def search_ann(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Approximate nearest-neighbor search via HNSW.

        Args:
            query: L2-normalized query vector of shape (embedding_dim,).
            k: Number of results to return.

        Returns:
            List of (image_id, similarity_score) tuples, descending by score.
        """
        if self._ann_index is None:
            self.build_ann_index()

        query = query.reshape(1, -1).astype(np.float32)
        scores, indices = self._ann_index.search(query, k)
        return [(self.image_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def search(self, query: np.ndarray, k: int = 5, mode: str = "knn") -> list[tuple[str, float]]:
        """Unified search interface.

        Args:
            query: L2-normalized query vector.
            k: Number of results.
            mode: "knn" for exact or "ann" for approximate.
        """
        if mode == "knn":
            return self.search_knn(query, k)
        elif mode == "ann":
            return self.search_ann(query, k)
        else:
            raise ValueError(f"Unknown search mode: {mode!r}. Use 'knn' or 'ann'.")

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self, directory: str | Path) -> None:
        """Save embeddings and image IDs to disk.

        Creates:
            directory/embeddings.npy  — float32 matrix
            directory/image_ids.json  — ordered list of IDs
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / "embeddings.npy", self.embeddings)

        with open(directory / "image_ids.json", "w") as f:
            json.dump(self.image_ids, f)

    def load(self, directory: str | Path) -> None:
        """Load embeddings and image IDs from disk, then rebuild indices."""
        directory = Path(directory)

        self.embeddings = np.load(directory / "embeddings.npy").astype(np.float32)
        self.embedding_dim = self.embeddings.shape[1]

        with open(directory / "image_ids.json") as f:
            self.image_ids = json.load(f)

        # Reset indices so they're rebuilt on first search
        self._knn_index = None
        self._ann_index = None

    @property
    def size(self) -> int:
        """Number of stored embeddings."""
        return len(self.image_ids)
