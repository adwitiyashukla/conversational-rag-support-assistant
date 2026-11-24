import hashlib
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from convrag.chunking import chunk_text, index_text
from convrag.config import Config
from convrag.embeddings import Embedder
from convrag.store import Store
from convrag.text import normalize_ws


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "head"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        if tag == "title":
            self.in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title":
            self.in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        elif not self.skip_depth:
            self.parts.append(data)


@dataclass
class LoadedDocument:
    title: str
    text: str


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        candidate = normalize_ws(line.lstrip("#").strip())
        if candidate:
            return candidate if len(candidate) <= 160 else fallback
    return fallback


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    fallback = path.stem.replace("_", " ").replace("-", " ")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(pages)
        meta_title = (reader.metadata or {}).get("/Title") if reader.metadata else None
        title = normalize_ws(str(meta_title)) if meta_title else _title_from_text(text, fallback)
        return LoadedDocument(title=title or fallback, text=text)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        parser = _HTMLText()
        parser.feed(raw)
        text = "".join(parser.parts)
        title = normalize_ws("".join(parser.title_parts)) or _title_from_text(text, fallback)
        return LoadedDocument(title=title, text=text)
    return LoadedDocument(title=_title_from_text(raw, fallback), text=raw)


def scan_folder(folder: Path, extensions: list[str], max_file_mb: int) -> list[Path]:
    allowed = {e.lower() for e in extensions}
    limit = max_file_mb * 1024 * 1024
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed and path.stat().st_size <= limit:
            files.append(path)
    return files


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class IngestReport:
    collection: str
    folder: str
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    chunks_embedded: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "collection": self.collection,
            "folder": self.folder,
            "added": self.added,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "removed": self.removed,
            "skipped": self.skipped,
            "chunks_embedded": self.chunks_embedded,
            "seconds": round(self.seconds, 3),
            "errors": self.errors,
        }


def ingest_document(
    store: Store,
    embedder: Embedder,
    cfg: Config,
    collection: str,
    rel_path: str,
    loaded: LoadedDocument,
    digest: str,
    size_bytes: int,
) -> int:
    chunks = chunk_text(loaded.text, cfg.chunking)
    texts = [index_text(loaded.title, c.text, cfg.chunking) for c in chunks]
    embeddings = embedder.embed_documents(texts) if texts else None
    rows = [(c.position, c.start, c.end, c.text, normalize_ws(c.text)) for c in chunks]
    store.replace_document(collection, rel_path, loaded.title, digest, size_bytes, rows, embeddings)
    return len(chunks)


def sync_folder(
    store: Store, embedder: Embedder, cfg: Config, folder: str | Path, collection: str
) -> IngestReport:
    started = time.perf_counter()
    root = Path(folder).resolve()
    report = IngestReport(collection=collection, folder=str(root))
    if not root.is_dir():
        raise FileNotFoundError(f"folder not found: {root}")
    existing = {d.path: d for d in store.list_documents(collection)}
    seen: set[str] = set()
    for path in scan_folder(root, cfg.ingest.extensions, cfg.ingest.max_file_mb):
        rel = path.relative_to(root).as_posix()
        seen.add(rel)
        data = path.read_bytes()
        digest = content_hash(data)
        current = existing.get(rel)
        if current and current.content_hash == digest:
            report.unchanged += 1
            continue
        try:
            loaded = load_document(path)
        except Exception as exc:
            report.skipped += 1
            report.errors.append(f"{rel}: {exc}")
            continue
        if not loaded.text.strip():
            report.skipped += 1
            continue
        count = ingest_document(store, embedder, cfg, collection, rel, loaded, digest, len(data))
        report.chunks_embedded += count
        if current:
            report.updated += 1
        else:
            report.added += 1
    for rel in existing:
        if rel not in seen and store.delete_document(collection, rel):
            report.removed += 1
    report.seconds = time.perf_counter() - started
    return report
