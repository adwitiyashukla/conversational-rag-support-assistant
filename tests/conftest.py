from pathlib import Path

import pytest

from convrag.config import Config, from_dict
from convrag.ingest import sync_folder
from convrag.runtime import Runtime

LICENSES = """# Renewing a driver license

You can renew your driver license online up to one year before it expires. Bring proof of identity
if you renew in person. A renewal costs 64 dollars and is valid for eight years.
Drivers over 70 must renew in person and pass a vision test.
"""

INSURANCE = """Insurance lapses

If your insurance ends, the DMV will suspend your registration and your license. You must get new
insurance within ten days or turn in your plates. A lapse letter explains how to clear the
suspension and how to pay the civil penalty.
"""

PARKING = """<html><head><title>Parking tickets</title></head>
<body><script>var x = 1;</script>
<p>Parking tickets must be paid within thirty days of the violation date.</p>
<p>You can dispute a parking ticket online with the ticket number and your plate.</p>
</body></html>
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "corpus"
    (folder / "dmv").mkdir(parents=True)
    (folder / "dmv" / "licenses.md").write_text(LICENSES, encoding="utf-8")
    (folder / "dmv" / "insurance.txt").write_text(INSURANCE, encoding="utf-8")
    (folder / "parking.html").write_text(PARKING, encoding="utf-8")
    (folder / "notes.csv").write_text("ignored,file\n", encoding="utf-8")
    return folder


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    raw = {
        "paths": {
            "db_file": str(tmp_path / "convrag.db"),
            "model_cache": str(tmp_path / "models"),
        },
        "chunking": {"max_words": 40, "min_words": 10, "overlap_sentences": 1},
        "embedding": {"backend": "hash", "hash_dim": 128},
        "reranker": {"enabled": True, "backend": "overlap", "candidates": 10, "min_score": -1.0},
        "retrieval": {"sparse_k": 20, "dense_k": 20, "top_k": 3},
        "llm": {"provider": "fake"},
        "server": {"trace_history": 5},
        "evaluation": {"results_dir": str(tmp_path / "results")},
    }
    return from_dict(raw, root=tmp_path)


@pytest.fixture
def runtime(cfg: Config, corpus: Path):
    rt = Runtime.build(cfg)
    sync_folder(rt.store, rt.embedder, cfg, corpus, "dmv")
    yield rt
    rt.close()
