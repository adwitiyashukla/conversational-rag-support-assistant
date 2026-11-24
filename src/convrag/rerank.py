from pathlib import Path
from typing import Protocol

from convrag.config import RerankerConfig
from convrag.text import content_tokens


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class OverlapReranker:
    name = "overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(content_tokens(query))
        if not query_tokens:
            return [0.0 for _ in texts]
        scores = []
        for text in texts:
            doc_tokens = set(content_tokens(text))
            shared = len(query_tokens & doc_tokens)
            scores.append(shared / len(query_tokens))
        return scores


class CrossEncoderReranker:
    def __init__(self, cfg: RerankerConfig, cache_dir: str | Path, threads: int):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.model = TextCrossEncoder(
            model_name=cfg.model, cache_dir=str(cache_dir), threads=threads
        )
        self.name = cfg.model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.rerank(query, texts)]


def build_reranker(cfg: RerankerConfig, cache_dir: str | Path, threads: int) -> Reranker | None:
    if cfg.backend == "none":
        return None
    if cfg.backend == "overlap":
        return OverlapReranker()
    return CrossEncoderReranker(cfg, cache_dir, threads)
