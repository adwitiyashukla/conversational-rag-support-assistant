import threading
from dataclasses import dataclass, field

import numpy as np

from convrag.config import RerankerConfig, RetrievalConfig
from convrag.embeddings import Embedder
from convrag.rerank import Reranker
from convrag.store import ChunkRecord, Store
from convrag.text import STOPWORDS, tokenize
from convrag.tracing import Trace


@dataclass
class Hit:
    chunk: ChunkRecord
    score: float
    sparse_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None
    queries: list[int] = field(default_factory=list)

    def to_dict(self, max_chars: int | None = None) -> dict:
        text = self.chunk.text
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "path": self.chunk.path,
            "title": self.chunk.title,
            "position": self.chunk.position,
            "start": self.chunk.start,
            "end": self.chunk.end,
            "score": round(self.score, 6),
            "sparse_rank": self.sparse_rank,
            "dense_rank": self.dense_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "text": text,
        }


def fts_match(query: str, max_tokens: int) -> str:
    seen: list[str] = []
    for token in tokenize(query):
        if token in STOPWORDS or token in seen:
            continue
        seen.append(token.replace("'", ""))
    seen = [t for t in seen if t]
    if len(seen) > max_tokens:
        seen = seen[-max_tokens:]
    return " OR ".join(f'"{t}"' for t in seen)


def rrf(rankings: list[list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


class VectorIndex:
    def __init__(self, store: Store):
        self.store = store
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}

    def get(self, collection: str) -> tuple[np.ndarray, np.ndarray]:
        version = self.store.index_version()
        with self.lock:
            cached = self.cache.get(collection)
            if cached and cached[0] == version:
                return cached[1], cached[2]
            ids, matrix = self.store.load_vectors(collection)
            self.cache[collection] = (version, ids, matrix)
            return ids, matrix

    def search(self, collection: str, query: np.ndarray, limit: int) -> list[tuple[int, float]]:
        ids, matrix = self.get(collection)
        if ids.size == 0:
            return []
        scores = matrix @ query.astype(np.float32)
        limit = min(limit, scores.shape[0])
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(int(ids[i]), float(scores[i])) for i in top]


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker | None,
        cfg: RetrievalConfig,
        rerank_cfg: RerankerConfig,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg
        self.rerank_cfg = rerank_cfg
        self.vectors = VectorIndex(store)

    def _sparse(self, query: str, collection: str, limit: int) -> list[int]:
        match = fts_match(query, self.cfg.max_query_tokens)
        return [h.chunk_id for h in self.store.sparse_search(match, collection, limit)]

    def _dense(self, queries: list[str], collection: str, limit: int) -> list[list[int]]:
        if not queries:
            return []
        vectors = self.embedder.embed_queries(queries)
        return [
            [cid for cid, _ in self.vectors.search(collection, vectors[i], limit)]
            for i in range(len(queries))
        ]

    def search(
        self,
        queries: list[str],
        collection: str,
        mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        rerank_query: str | None = None,
        trace: Trace | None = None,
    ) -> list[Hit]:
        mode = mode or self.cfg.mode
        top_k = top_k or self.cfg.top_k
        wanted = self.rerank_cfg.enabled if rerank is None else rerank
        use_rerank = wanted and self.reranker is not None
        trace = trace or Trace(kind="search")
        queries = [q for q in queries if q.strip()]
        if not queries:
            return []
        rerank_query = rerank_query or queries[0]
        rankings: list[list[int]] = []
        sparse_rank: dict[int, int] = {}
        dense_rank: dict[int, int] = {}
        query_sources: dict[int, set[int]] = {}
        if mode in ("bm25", "hybrid"):
            with trace.span("sparse", queries=len(queries)):
                for qi, query in enumerate(queries):
                    ranking = self._sparse(query, collection, self.cfg.sparse_k)
                    rankings.append(ranking)
                    for rank, cid in enumerate(ranking, start=1):
                        sparse_rank.setdefault(cid, rank)
                        query_sources.setdefault(cid, set()).add(qi)
        if mode in ("dense", "hybrid"):
            with trace.span("dense", queries=len(queries)):
                for qi, ranking in enumerate(self._dense(queries, collection, self.cfg.dense_k)):
                    rankings.append(ranking)
                    for rank, cid in enumerate(ranking, start=1):
                        dense_rank.setdefault(cid, rank)
                        query_sources.setdefault(cid, set()).add(qi)
        with trace.span("fuse", lists=len(rankings)):
            if len(rankings) == 1:
                fused = {cid: 1.0 / (self.cfg.rrf_k + r) for r, cid in enumerate(rankings[0], 1)}
            else:
                fused = rrf(rankings, self.cfg.rrf_k)
            ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        pool = self.rerank_cfg.candidates if use_rerank else top_k
        candidates = ordered[:pool]
        records = self.store.get_chunks([cid for cid, _ in candidates])
        hits = [
            Hit(
                chunk=records[cid],
                score=score,
                sparse_rank=sparse_rank.get(cid),
                dense_rank=dense_rank.get(cid),
                queries=sorted(query_sources.get(cid, set())),
            )
            for cid, score in candidates
            if cid in records
        ]
        if use_rerank and hits:
            with trace.span("rerank", candidates=len(hits)):
                passages = [f"{h.chunk.title}\n{h.chunk.text}" for h in hits]
                scores = self.reranker.score(rerank_query, passages)
                for hit, score in zip(hits, scores, strict=True):
                    hit.rerank_score = score
                hits.sort(key=lambda h: (-(h.rerank_score or 0.0), -h.score))
        return hits[:top_k]
