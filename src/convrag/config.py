import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ENV_PREFIX = "CONVRAG_"


class ConfigError(ValueError):
    pass


@dataclass
class PathsConfig:
    db_file: str = "data/convrag.db"
    model_cache: str = "data/models"


@dataclass
class IngestConfig:
    extensions: list[str] = field(default_factory=lambda: [".md", ".txt", ".html", ".htm", ".pdf"])
    max_file_mb: int = 25


@dataclass
class ChunkingConfig:
    max_words: int = 180
    min_words: int = 40
    overlap_sentences: int = 1
    title_prefix: bool = True


@dataclass
class EmbeddingConfig:
    backend: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = ""
    batch_size: int = 32
    threads: int = 4
    hash_dim: int = 256


@dataclass
class RerankerConfig:
    enabled: bool = False
    backend: str = "fastembed"
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    query_source: str = "message"
    candidates: int = 20
    min_score: float = -8.0


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"
    sparse_k: int = 50
    dense_k: int = 50
    rrf_k: int = 60
    top_k: int = 5
    max_query_tokens: int = 64


@dataclass
class RewriteConfig:
    strategy: str = "window"
    window_turns: int = 3
    history_max_chars: int = 1500


@dataclass
class LLMConfig:
    provider: str = "openai"
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:3b"
    api_key_env: str = "CONVRAG_LLM_API_KEY"
    timeout_seconds: float = 300.0
    max_tokens: int = 200
    temperature: float = 0.0

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class GenerationConfig:
    history_turns: int = 6
    min_citation_overlap: float = 0.35
    abstain_message: str = "I could not find this in the documentation."


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    trace_history: int = 500
    api_key_env: str = "CONVRAG_API_KEY"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class EvaluationConfig:
    raw_dir: str = "data/raw/multidoc2dial"
    corpus_dir: str = "data/corpus/multidoc2dial"
    collection: str = "multidoc2dial"
    results_dir: str = "data/results"
    generation_sample: int = 300
    seed: int = 7


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    root: Path = field(default_factory=Path.cwd)

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.root / path


SECTIONS = {
    "paths": PathsConfig,
    "ingest": IngestConfig,
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "reranker": RerankerConfig,
    "retrieval": RetrievalConfig,
    "rewrite": RewriteConfig,
    "llm": LLMConfig,
    "generation": GenerationConfig,
    "server": ServerConfig,
    "evaluation": EvaluationConfig,
}

CHOICES = {
    ("embedding", "backend"): {"fastembed", "hash"},
    ("reranker", "backend"): {"fastembed", "overlap", "none"},
    ("reranker", "query_source"): {"query", "message"},
    ("retrieval", "mode"): {"bm25", "dense", "hybrid"},
    ("rewrite", "strategy"): {"last_turn", "window", "history", "fusion", "rewrite"},
    ("llm", "provider"): {"openai", "anthropic", "fake"},
}

POSITIVE = {
    ("ingest", "max_file_mb"),
    ("chunking", "max_words"),
    ("chunking", "min_words"),
    ("embedding", "batch_size"),
    ("embedding", "threads"),
    ("embedding", "hash_dim"),
    ("reranker", "candidates"),
    ("retrieval", "sparse_k"),
    ("retrieval", "dense_k"),
    ("retrieval", "rrf_k"),
    ("retrieval", "top_k"),
    ("retrieval", "max_query_tokens"),
    ("rewrite", "window_turns"),
    ("rewrite", "history_max_chars"),
    ("llm", "timeout_seconds"),
    ("llm", "max_tokens"),
    ("generation", "history_turns"),
    ("server", "port"),
    ("server", "trace_history"),
    ("evaluation", "generation_sample"),
}


def _build_section(name: str, cls: type, raw: dict) -> object:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"[{name}] has unknown keys: {', '.join(unknown)}")
    values = {}
    for key, value in raw.items():
        expected = known[key].type
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if expected in (int, float, str, bool):
            ok = isinstance(value, expected) and not (
                expected is not bool and isinstance(value, bool)
            )
            if not ok:
                raise ConfigError(f"[{name}].{key} must be {expected.__name__}")
        elif expected == list[str]:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ConfigError(f"[{name}].{key} must be a list of strings")
        values[key] = value
    return cls(**values)


def validate(cfg: Config) -> Config:
    for (section, key), allowed in CHOICES.items():
        value = getattr(getattr(cfg, section), key)
        if value not in allowed:
            raise ConfigError(f"[{section}].{key} must be one of {sorted(allowed)}, got {value!r}")
    for section, key in POSITIVE:
        value = getattr(getattr(cfg, section), key)
        if value <= 0:
            raise ConfigError(f"[{section}].{key} must be positive")
    if cfg.chunking.min_words > cfg.chunking.max_words:
        raise ConfigError("[chunking].min_words cannot exceed max_words")
    if cfg.chunking.overlap_sentences < 0:
        raise ConfigError("[chunking].overlap_sentences cannot be negative")
    if not 0.0 <= cfg.generation.min_citation_overlap <= 1.0:
        raise ConfigError("[generation].min_citation_overlap must be between 0 and 1")
    if not 0.0 <= cfg.llm.temperature <= 2.0:
        raise ConfigError("[llm].temperature must be between 0 and 2")
    if cfg.retrieval.top_k > max(cfg.retrieval.sparse_k, cfg.retrieval.dense_k):
        raise ConfigError("[retrieval].top_k cannot exceed the candidate pool")
    if not cfg.ingest.extensions or not all(e.startswith(".") for e in cfg.ingest.extensions):
        raise ConfigError("[ingest].extensions must be a non-empty list of dotted extensions")
    return cfg


def _parse_env(name: str, key: str, expected: type, value: str):
    if expected is bool:
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{ENV_PREFIX}{name.upper()}_{key.upper()} must be true or false")
    if expected in (int, float):
        try:
            return expected(value)
        except ValueError as exc:
            raise ConfigError(
                f"{ENV_PREFIX}{name.upper()}_{key.upper()} must be {expected.__name__}"
            ) from exc
    if expected == list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def apply_env(raw: dict, environ: dict | None = None) -> dict:
    environ = os.environ if environ is None else environ
    merged = {name: dict(raw.get(name, {})) for name in SECTIONS}
    for name, cls in SECTIONS.items():
        for field_info in fields(cls):
            variable = f"{ENV_PREFIX}{name.upper()}_{field_info.name.upper()}"
            if variable in environ:
                merged[name][field_info.name] = _parse_env(
                    name, field_info.name, field_info.type, environ[variable]
                )
    return merged


def from_dict(raw: dict, root: Path | None = None, environ: dict | None = None) -> Config:
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ConfigError(f"unknown config sections: {', '.join(unknown)}")
    for name in SECTIONS:
        if name in raw and not isinstance(raw[name], dict):
            raise ConfigError(f"[{name}] must be a table")
    sections = {
        name: _build_section(name, cls, body)
        for (name, cls), body in zip(
            SECTIONS.items(), apply_env(raw, environ).values(), strict=True
        )
    }
    return validate(Config(root=root or Path.cwd(), **sections))


def load(path: str | Path = "config.toml") -> Config:
    file = Path(path)
    if not file.exists():
        raise ConfigError(f"config file not found: {file}")
    with file.open("rb") as handle:
        try:
            raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"could not parse {file}: {exc}") from exc
    return from_dict(raw, root=file.resolve().parent)
