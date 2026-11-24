import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from convrag.config import Config
from convrag.ingest import sync_folder
from convrag.llm import LLMError
from convrag.runtime import Runtime
from convrag.tracing import Trace

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    collection: str = Field(min_length=1, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: str | None = None
    rerank: bool | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    collection: str = Field(min_length=1, max_length=200)
    session_id: str | None = Field(default=None, max_length=64)
    stream: bool = False


class IngestRequest(BaseModel):
    folder: str = Field(min_length=1)
    collection: str = Field(min_length=1, max_length=200)


def _sse(events: Iterator[dict]) -> Iterator[str]:
    try:
        for event in events:
            kind = event.pop("type")
            yield f"event: {kind}\ndata: {json.dumps(event)}\n\n"
    except LLMError as exc:
        yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"


def create_app(cfg: Config, runtime: Runtime | None = None) -> FastAPI:
    state = {"runtime": runtime}

    def get_runtime() -> Runtime:
        if state["runtime"] is None:
            state["runtime"] = Runtime.build(cfg)
        return state["runtime"]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        get_runtime()
        yield

    app = FastAPI(
        title="convrag", version="0.1.0", docs_url="/docs", redoc_url=None, lifespan=lifespan
    )

    def require_key(request: Request) -> None:
        expected = cfg.server.api_key
        if not expected:
            return
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        token = token or request.headers.get("x-api-key", "")
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid api key")

    protected = [Depends(require_key)]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        rt = get_runtime()
        return {
            "status": "ready",
            "embedding_model": rt.embedder.name,
            "embedding_dim": rt.embedder.dim,
            "reranker": rt.reranker.name if rt.reranker else None,
            "llm": rt.llm.name,
            "chunks": rt.store.chunk_count(),
        }

    @app.get("/collections", dependencies=protected)
    def collections() -> dict:
        return {"collections": get_runtime().store.collections()}

    @app.post("/ingest", dependencies=protected)
    def ingest(body: IngestRequest) -> dict:
        rt = get_runtime()
        folder = cfg.resolve(body.folder)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"folder not found: {body.folder}")
        return sync_folder(rt.store, rt.embedder, cfg, folder, body.collection).to_dict()

    @app.post("/search", dependencies=protected)
    def search(body: SearchRequest) -> dict:
        rt = get_runtime()
        if body.mode is not None and body.mode not in ("bm25", "dense", "hybrid"):
            raise HTTPException(status_code=400, detail="mode must be bm25, dense or hybrid")
        trace = Trace(kind="search")
        hits = rt.retriever.search(
            [body.query],
            body.collection,
            mode=body.mode,
            top_k=body.top_k,
            rerank=body.rerank,
            trace=trace,
        )
        rt.metrics.record_trace(trace)
        return {"hits": [h.to_dict() for h in hits], "trace": trace.to_dict()}

    @app.post("/chat", dependencies=protected)
    def chat(body: ChatRequest):
        rt = get_runtime()
        if body.stream:
            events = rt.chat.respond_stream(body.message, body.collection, body.session_id)
            return StreamingResponse(
                _sse(events),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = rt.chat.respond(body.message, body.collection, body.session_id)
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"session_id": result.session_id, **result.payload()}

    @app.get("/sessions/{session_id}", dependencies=protected)
    def session(session_id: str) -> dict:
        rt = get_runtime()
        record = rt.store.get_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session not found")
        messages = rt.store.get_messages(session_id)
        return {
            **record,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                    "payload": m.payload,
                }
                for m in messages
            ],
        }

    @app.get("/traces", dependencies=protected)
    def traces(limit: int = 20) -> dict:
        limit = max(1, min(limit, 200))
        return {"traces": get_runtime().store.recent_traces(limit)}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return get_runtime().metrics.render()

    return app
