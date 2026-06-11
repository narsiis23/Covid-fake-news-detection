"""LaBSE sentence embedding utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from src.utils import ensure_dir, resolve_path

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from exc


class LaBSEEmbedder:
    """Wrapper around the LaBSE SentenceTransformer model."""

    def __init__(self, model_name: str, batch_size: int = 32, show_progress_bar: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into dense embeddings."""
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_sklearn_transformer(self, text_column: str) -> ColumnTransformer:
        """Build a sklearn ColumnTransformer for the text column."""

        def _encode_batch(items: np.ndarray) -> np.ndarray:
            texts = [str(item) for item in np.ravel(items)]
            return self.encode(texts)

        embedder = FunctionTransformer(_encode_batch, validate=False)
        return ColumnTransformer(
            transformers=[("embedder", embedder, text_column)],
            remainder="drop",
        )


def build_embedder(config: dict) -> LaBSEEmbedder:
    """Create an embedder from configuration."""
    embed_cfg = config["embedding"]
    return LaBSEEmbedder(
        model_name=embed_cfg["model_name"],
        batch_size=embed_cfg.get("batch_size", 32),
        show_progress_bar=embed_cfg.get("show_progress_bar", True),
    )


def embedding_cache_path(config: dict, split_name: str) -> Path:
    """Return the cache path for precomputed embeddings."""
    cache_dir = resolve_path(config["paths"]["embeddings_cache"], config)
    ensure_dir(cache_dir)
    return cache_dir / f"{split_name}.npy"


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """Save embeddings to disk."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    np.save(output_path, embeddings)


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from disk."""
    return np.load(path)
