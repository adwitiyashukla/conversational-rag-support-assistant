from pathlib import Path

import pytest

from convrag.config import ConfigError, from_dict, load

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_loads_and_resolves_paths():
    cfg = load(ROOT / "config.toml")
    assert cfg.embedding.backend == "fastembed"
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.resolve(cfg.paths.db_file) == ROOT / "data" / "convrag.db"
    assert cfg.rewrite.strategy in ("last_turn", "window", "history", "fusion", "rewrite")


def test_defaults_apply_for_missing_sections():
    cfg = from_dict({})
    assert cfg.chunking.max_words == 180
    assert cfg.llm.provider == "openai"
    assert cfg.llm.api_key == ""


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ({"nonsense": {}}, "unknown config sections"),
        ({"chunking": {"max_words": "big"}}, "must be int"),
        ({"chunking": {"max_words": 10, "min_words": 20}}, "min_words cannot exceed"),
        ({"chunking": {"overlap_sentences": -1}}, "cannot be negative"),
        ({"retrieval": {"mode": "magic"}}, "must be one of"),
        ({"retrieval": {"top_k": 0}}, "must be positive"),
        ({"retrieval": {"top_k": 500}}, "candidate pool"),
        ({"generation": {"min_citation_overlap": 1.5}}, "between 0 and 1"),
        ({"llm": {"temperature": 3.0}}, "between 0 and 2"),
        ({"ingest": {"extensions": ["md"]}}, "dotted extensions"),
        ({"ingest": {"extensions": "md"}}, "list of strings"),
        ({"embedding": {"threads": True}}, "must be int"),
        ({"server": {"unknown_key": 1}}, "unknown keys"),
        ({"paths": "data"}, "must be a table"),
    ],
)
def test_invalid_values_raise_config_errors(raw, fragment):
    with pytest.raises(ConfigError) as info:
        from_dict(raw)
    assert fragment in str(info.value)


def test_missing_or_broken_file(tmp_path: Path):
    with pytest.raises(ConfigError):
        load(tmp_path / "nope.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("[paths\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load(broken)


def test_env_backed_keys(monkeypatch):
    cfg = from_dict({"llm": {"api_key_env": "TEST_LLM_KEY"}})
    monkeypatch.setenv("TEST_LLM_KEY", "abc")
    assert cfg.llm.api_key == "abc"


def test_environment_overrides_config_values():
    environ = {
        "CONVRAG_LLM_BASE_URL": "http://ollama:11434/v1",
        "CONVRAG_RETRIEVAL_TOP_K": "7",
        "CONVRAG_CHUNKING_TITLE_PREFIX": "false",
        "CONVRAG_INGEST_EXTENSIONS": ".md, .txt",
        "CONVRAG_LLM_TEMPERATURE": "0.5",
        "UNRELATED": "x",
    }
    cfg = from_dict({"retrieval": {"top_k": 3}}, environ=environ)
    assert cfg.llm.base_url == "http://ollama:11434/v1"
    assert cfg.retrieval.top_k == 7
    assert cfg.chunking.title_prefix is False
    assert cfg.ingest.extensions == [".md", ".txt"]
    assert cfg.llm.temperature == 0.5
    with pytest.raises(ConfigError, match="must be int"):
        from_dict({}, environ={"CONVRAG_RETRIEVAL_TOP_K": "seven"})
    with pytest.raises(ConfigError, match="true or false"):
        from_dict({}, environ={"CONVRAG_CHUNKING_TITLE_PREFIX": "maybe"})
