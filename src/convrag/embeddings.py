import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from convrag.config import EmbeddingConfig
from convrag.text import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            bucket = value % self.dim
            sign = 1.0 if (value >> 40) & 1 else -1.0
            vector[bucket] += sign
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize_rows(np.stack([self._embed(t) for t in texts]))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


class FastEmbedEmbedder:
    def __init__(self, cfg: EmbeddingConfig, cache_dir: str | Path):
        from fastembed import TextEmbedding

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(
            model_name=cfg.model, cache_dir=str(cache_dir), threads=cfg.threads
        )
        self.name = cfg.model
        self.query_prefix = cfg.query_prefix
        self.batch_size = cfg.batch_size
        self.dim = int(self.embed_documents(["dimension probe"]).shape[1])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        vectors = list(self.model.embed(texts, batch_size=self.batch_size))
        return normalize_rows(np.stack(vectors))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents([f"{self.query_prefix}{t}" for t in texts])


def build_embedder(cfg: EmbeddingConfig, cache_dir: str | Path) -> Embedder:
    if cfg.backend == "hash":
        return HashEmbedder(cfg.hash_dim)
    return FastEmbedEmbedder(cfg, cache_dir)
