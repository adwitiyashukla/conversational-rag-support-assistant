import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    collection TEXT NOT NULL,
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE(collection, path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    position INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, body, collection UNINDEXED,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    session_id TEXT,
    payload TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    collection: str
    path: str
    title: str
    content_hash: str
    size_bytes: int
    chunk_count: int
    ingested_at: str


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    document_id: int
    collection: str
    position: int
    start: int
    end: int
    text: str
    path: str
    title: str


@dataclass(frozen=True)
class SparseHit:
    chunk_id: int
    score: float


@dataclass
class Message:
    role: str
    content: str
    payload: dict | None = None
    created_at: str = ""


class Store:
    def __init__(self, db_file: str | Path):
        self.path = Path(db_file)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('index_version', '0')")
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def index_version(self) -> int:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key='index_version'").fetchone()
        return int(row["value"])

    def bump_index_version(self) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
                "WHERE key='index_version'"
            )

    def get_document(self, collection: str, path: str) -> DocumentRecord | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM documents WHERE collection=? AND path=?", (collection, path)
            ).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def list_documents(self, collection: str) -> list[DocumentRecord]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM documents WHERE collection=? ORDER BY path", (collection,)
            ).fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]

    def replace_document(
        self,
        collection: str,
        path: str,
        title: str,
        content_hash: str,
        size_bytes: int,
        chunks: list[tuple[int, int, int, str, str]],
        embeddings: np.ndarray | None,
    ) -> int:
        with self.lock:
            existing = self.conn.execute(
                "SELECT id FROM documents WHERE collection=? AND path=?", (collection, path)
            ).fetchone()
            if existing:
                self._delete_document_rows(existing["id"])
            cursor = self.conn.execute(
                "INSERT INTO documents(collection, path, title, content_hash, size_bytes, "
                "chunk_count, ingested_at) VALUES(?,?,?,?,?,?,?)",
                (collection, path, title, content_hash, size_bytes, len(chunks), utcnow()),
            )
            document_id = cursor.lastrowid
            for index, (position, start, end, text, fts_body) in enumerate(chunks):
                blob = None
                if embeddings is not None:
                    blob = np.ascontiguousarray(embeddings[index], dtype=np.float32).tobytes()
                chunk_cursor = self.conn.execute(
                    "INSERT INTO chunks(document_id, collection, position, start_char, end_char, "
                    "text, embedding) VALUES(?,?,?,?,?,?,?)",
                    (document_id, collection, position, start, end, text, blob),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, title, body, collection) VALUES(?,?,?,?)",
                    (chunk_cursor.lastrowid, title, fts_body, collection),
                )
            self.bump_index_version()
            self.conn.commit()
            return int(document_id)

    def _delete_document_rows(self, document_id: int) -> None:
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE document_id=?", (document_id,)
        ).fetchall()
        for row in rows:
            self.conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row["id"],))
        self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        self.conn.execute("DELETE FROM documents WHERE id=?", (document_id,))

    def delete_document(self, collection: str, path: str) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT id FROM documents WHERE collection=? AND path=?", (collection, path)
            ).fetchone()
            if not row:
                return False
            self._delete_document_rows(row["id"])
            self.bump_index_version()
            self.conn.commit()
            return True

    def collections(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT collection, COUNT(*) AS documents, SUM(chunk_count) AS chunks "
                "FROM documents GROUP BY collection ORDER BY collection"
            ).fetchall()
        return [dict(r) for r in rows]

    def sparse_search(self, match: str, collection: str, limit: int) -> list[SparseHit]:
        if not match:
            return []
        with self.lock:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts, 2.0, 1.0) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH ? AND collection = ? ORDER BY score LIMIT ?",
                (match, collection, limit),
            ).fetchall()
        return [SparseHit(chunk_id=int(r["rowid"]), score=-float(r["score"])) for r in rows]

    def load_vectors(self, collection: str) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, embedding FROM chunks WHERE collection=? AND embedding IS NOT NULL "
                "ORDER BY id",
                (collection,),
            ).fetchall()
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        ids = np.array([r["id"] for r in rows], dtype=np.int64)
        matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return ids, matrix

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, ChunkRecord]:
        wanted = [int(i) for i in chunk_ids]
        if not wanted:
            return {}
        placeholders = ",".join("?" for _ in wanted)
        with self.lock:
            rows = self.conn.execute(
                f"SELECT c.id, c.document_id, c.collection, c.position, c.start_char, "
                f"c.end_char, c.text, d.path, d.title FROM chunks c "
                f"JOIN documents d ON d.id = c.document_id WHERE c.id IN ({placeholders})",
                wanted,
            ).fetchall()
        return {
            int(r["id"]): ChunkRecord(
                id=int(r["id"]),
                document_id=int(r["document_id"]),
                collection=r["collection"],
                position=int(r["position"]),
                start=int(r["start_char"]),
                end=int(r["end_char"]),
                text=r["text"],
                path=r["path"],
                title=r["title"],
            )
            for r in rows
        }

    def chunk_count(self, collection: str | None = None) -> int:
        with self.lock:
            if collection is None:
                row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            else:
                row = self.conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE collection=?", (collection,)
                ).fetchone()
        return int(row["n"])

    def create_session(self, collection: str, session_id: str | None = None) -> str:
        session_id = session_id or uuid.uuid4().hex
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions(id, collection, created_at) VALUES(?,?,?)",
                (session_id, collection, utcnow()),
            )
            self.conn.commit()
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def add_message(
        self, session_id: str, role: str, content: str, payload: dict | None = None
    ) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO messages(session_id, role, content, payload, created_at) "
                "VALUES(?,?,?,?,?)",
                (session_id, role, content, json.dumps(payload) if payload else None, utcnow()),
            )
            self.conn.commit()

    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        with self.lock:
            if limit is None:
                rows = self.conn.execute(
                    "SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM (SELECT * FROM messages WHERE session_id=? "
                    "ORDER BY id DESC LIMIT ?) ORDER BY id",
                    (session_id, limit),
                ).fetchall()
        return [
            Message(
                role=r["role"],
                content=r["content"],
                payload=json.loads(r["payload"]) if r["payload"] else None,
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def add_trace(self, kind: str, session_id: str | None, payload: dict, keep: int) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO traces(created_at, kind, session_id, payload) VALUES(?,?,?,?)",
                (utcnow(), kind, session_id, json.dumps(payload)),
            )
            self.conn.execute(
                "DELETE FROM traces WHERE id NOT IN "
                "(SELECT id FROM traces ORDER BY id DESC LIMIT ?)",
                (keep,),
            )
            self.conn.commit()

    def recent_traces(self, limit: int) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, created_at, kind, session_id, payload FROM traces "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "kind": r["kind"],
                "session_id": r["session_id"],
                **json.loads(r["payload"]),
            }
            for r in rows
        ]
