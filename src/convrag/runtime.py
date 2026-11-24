from dataclasses import dataclass

from convrag.chat import ChatService
from convrag.config import Config
from convrag.embeddings import Embedder, build_embedder
from convrag.llm import LLM, build_llm
from convrag.rerank import Reranker, build_reranker
from convrag.retrieval import Retriever
from convrag.store import Store
from convrag.tracing import Metrics


@dataclass
class Runtime:
    cfg: Config
    store: Store
    embedder: Embedder
    reranker: Reranker | None
    retriever: Retriever
    llm: LLM
    chat: ChatService
    metrics: Metrics

    @classmethod
    def build(cls, cfg: Config, llm: LLM | None = None) -> "Runtime":
        store = Store(cfg.resolve(cfg.paths.db_file))
        cache = cfg.resolve(cfg.paths.model_cache)
        embedder = build_embedder(cfg.embedding, cache)
        reranker = build_reranker(cfg.reranker, cache, cfg.embedding.threads)
        retriever = Retriever(store, embedder, reranker, cfg.retrieval, cfg.reranker)
        llm = llm or build_llm(cfg.llm)
        metrics = Metrics()
        chat = ChatService(store, retriever, llm, cfg, metrics)
        return cls(cfg, store, embedder, reranker, retriever, llm, chat, metrics)

    def close(self) -> None:
        self.store.close()
