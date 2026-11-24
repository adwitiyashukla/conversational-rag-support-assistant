from collections.abc import Iterator
from dataclasses import dataclass, field

from convrag.citations import VerifiedAnswer, verify
from convrag.config import Config
from convrag.llm import LLM, LLMError
from convrag.prompts import (
    ABSTAIN_PATTERN,
    ANSWER_SYSTEM,
    ANSWER_USER,
    format_conversation,
    format_sources,
)
from convrag.retrieval import Hit, Retriever
from convrag.rewrite import Formulation, formulate
from convrag.store import Store
from convrag.text import normalize_ws
from convrag.tracing import Metrics, Trace


@dataclass
class ChatResult:
    message: str
    answer: str
    raw_answer: str
    abstained: bool
    abstain_reason: str | None
    grounded: bool
    formulation: Formulation
    hits: list[Hit]
    verified: VerifiedAnswer
    trace: Trace
    session_id: str | None = None
    completion_tokens: int = 0
    prompt_tokens: int = 0

    def payload(self) -> dict:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "grounded": self.grounded,
            "strategy": self.formulation.strategy,
            "queries": self.formulation.queries,
            "rewrite_fallback": self.formulation.fallback,
            "sources": [h.to_dict(max_chars=600) for h in self.hits],
            "citations": self.verified.to_dict()["citations"],
            "cited_sources": self.verified.cited_sources,
            "trace": self.trace.to_dict(),
        }


@dataclass
class ChatService:
    store: Store
    retriever: Retriever
    llm: LLM
    cfg: Config
    metrics: Metrics = field(default_factory=Metrics)

    def history(self, session_id: str) -> list[tuple[str, str]]:
        limit = self.cfg.generation.history_turns
        return [(m.role, m.content) for m in self.store.get_messages(session_id, limit)]

    def run(
        self,
        history: list[tuple[str, str]],
        message: str,
        collection: str,
        strategy: str | None = None,
        stream: bool = True,
    ) -> Iterator[dict]:
        trace = Trace(kind="chat")
        message = normalize_ws(message)
        with trace.span("rewrite") as span:
            formulation = formulate(history, message, self.cfg.rewrite, self.llm, strategy)
            span.meta.update(strategy=formulation.strategy, fallback=formulation.fallback)
        yield {
            "type": "queries",
            "strategy": formulation.strategy,
            "queries": formulation.queries,
            "fallback": formulation.fallback,
        }
        rerank_query = formulation.rerank_query
        if self.cfg.reranker.query_source == "message":
            rerank_query = message
        hits = self.retriever.search(
            formulation.queries,
            collection,
            rerank_query=rerank_query,
            trace=trace,
        )
        yield {"type": "sources", "sources": [h.to_dict(max_chars=400) for h in hits]}
        abstain_reason = None
        if not hits:
            abstain_reason = "no_results"
        elif (
            hits[0].rerank_score is not None and hits[0].rerank_score < self.cfg.reranker.min_score
        ):
            abstain_reason = "low_relevance"
        raw = ""
        prompt_tokens = completion_tokens = 0
        sources = [(h.chunk.title, normalize_ws(h.chunk.text)) for h in hits]
        if abstain_reason is None:
            recent = history[-self.cfg.generation.history_turns :]
            user_prompt = ANSWER_USER.format(
                sources=format_sources(sources),
                conversation=format_conversation(recent),
                message=message,
            )
            messages = [{"role": "user", "content": user_prompt}]
            with trace.span("generate", model=self.llm.name) as span:
                if stream:
                    parts = []
                    for piece in self.llm.stream(ANSWER_SYSTEM, messages):
                        parts.append(piece)
                        yield {"type": "token", "text": piece}
                    raw = "".join(parts).strip()
                else:
                    completion = self.llm.complete(ANSWER_SYSTEM, messages)
                    raw = completion.text
                    prompt_tokens = completion.prompt_tokens
                    completion_tokens = completion.completion_tokens
                span.meta.update(chars=len(raw))
            if not raw or ABSTAIN_PATTERN.search(raw):
                abstain_reason = "model_abstained"
        if abstain_reason is None:
            with trace.span("verify"):
                verified = verify(
                    raw, [text for _, text in sources], self.cfg.generation.min_citation_overlap
                )
            if not verified.text:
                abstain_reason = "empty_answer"
        if abstain_reason is not None:
            verified = VerifiedAnswer(text=self.cfg.generation.abstain_message)
            self.metrics.inc("convrag_abstentions_total", reason=abstain_reason)
        trace.meta.update(
            strategy=formulation.strategy,
            hits=len(hits),
            abstained=abstain_reason is not None,
            grounded=verified.grounded,
        )
        self.metrics.record_trace(trace)
        result = ChatResult(
            message=message,
            answer=verified.text,
            raw_answer=raw,
            abstained=abstain_reason is not None,
            abstain_reason=abstain_reason,
            grounded=verified.grounded,
            formulation=formulation,
            hits=hits,
            verified=verified,
            trace=trace,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        yield {"type": "done", "result": result}

    def answer(
        self,
        history: list[tuple[str, str]],
        message: str,
        collection: str,
        strategy: str | None = None,
    ) -> ChatResult:
        result = None
        for event in self.run(history, message, collection, strategy, stream=False):
            if event["type"] == "done":
                result = event["result"]
        if result is None:
            raise LLMError("chat produced no result")
        return result

    def _persist(self, session_id: str, result: ChatResult) -> None:
        result.session_id = session_id
        self.store.add_message(session_id, "user", result.message)
        self.store.add_message(session_id, "assistant", result.answer, result.payload())
        self.store.add_trace(
            "chat", session_id, result.trace.to_dict(), self.cfg.server.trace_history
        )

    def respond(self, message: str, collection: str, session_id: str | None = None) -> ChatResult:
        session_id = self.store.create_session(collection, session_id)
        result = self.answer(self.history(session_id), message, collection)
        self._persist(session_id, result)
        return result

    def respond_stream(
        self, message: str, collection: str, session_id: str | None = None
    ) -> Iterator[dict]:
        session_id = self.store.create_session(collection, session_id)
        yield {"type": "start", "session_id": session_id}
        for event in self.run(self.history(session_id), message, collection, stream=True):
            if event["type"] == "done":
                result = event["result"]
                self._persist(session_id, result)
                yield {"type": "done", "session_id": session_id, **result.payload()}
            else:
                yield event
