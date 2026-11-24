import json
from pathlib import Path

import pytest

from convrag import evaluation
from convrag.ingest import sync_folder
from convrag.retrieval import Hit
from convrag.store import ChunkRecord

INSURANCE_TEXT = (
    "\n\nInsurance lapses \n"
    "If your insurance ends, the DMV will suspend your registration and your license. "
    "You must get new insurance within ten days or turn in your plates. "
    "A lapse letter explains how to clear the suspension.\n"
)
PARKING_TEXT = (
    "\n\nParking tickets \n"
    "Parking tickets must be paid within thirty days of the violation date. "
    "You can dispute a parking ticket online with the ticket number and your plate.\n"
)


def spans_for(text: str, sentences: list[str]) -> dict:
    spans = {}
    for index, sentence in enumerate(sentences, start=1):
        start = text.index(sentence)
        spans[str(index)] = {
            "id_sp": str(index),
            "start_sp": start,
            "end_sp": start + len(sentence),
            "text_sp": sentence,
        }
    return spans


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    docs = {
        "doc_data": {
            "dmv": {
                "Insurance lapses#1_0": {
                    "title": "Insurance lapses#1",
                    "doc_id": "Insurance lapses#1_0",
                    "domain": "dmv",
                    "doc_text": INSURANCE_TEXT,
                    "spans": spans_for(
                        INSURANCE_TEXT,
                        [
                            "If your insurance ends, the DMV will suspend your registration and "
                            "your license.",
                            "You must get new insurance within ten days or turn in your plates.",
                            "A lapse letter explains how to clear the suspension.",
                        ],
                    ),
                },
                "Parking tickets#1_0": {
                    "title": "Parking tickets#1",
                    "doc_id": "Parking tickets#1_0",
                    "domain": "dmv",
                    "doc_text": PARKING_TEXT,
                    "spans": spans_for(
                        PARKING_TEXT,
                        [
                            "Parking tickets must be paid within thirty days of the violation "
                            "date.",
                            "You can dispute a parking ticket online with the ticket number and "
                            "your plate.",
                        ],
                    ),
                },
            }
        }
    }
    dialogs = {
        "dial_data": {
            "dmv": [
                {
                    "dial_id": "d1",
                    "turns": [
                        {
                            "turn_id": 1,
                            "role": "user",
                            "da": "query_condition",
                            "references": [{"id_sp": "1", "doc_id": "Insurance lapses#1_0"}],
                            "utterance": "My insurance ended so what should I do",
                        },
                        {
                            "turn_id": 2,
                            "role": "agent",
                            "da": "respond_solution",
                            "references": [{"id_sp": "1", "doc_id": "Insurance lapses#1_0"}],
                            "utterance": "We will suspend your registration and license",
                        },
                        {
                            "turn_id": 3,
                            "role": "user",
                            "da": "query_solution",
                            "references": [],
                            "utterance": "How long do I have",
                        },
                        {
                            "turn_id": 4,
                            "role": "agent",
                            "da": "respond_solution",
                            "references": [{"id_sp": "2", "doc_id": "Insurance lapses#1_0"}],
                            "utterance": "You have ten days to get new insurance",
                        },
                        {
                            "turn_id": 5,
                            "role": "user",
                            "da": "query_solution",
                            "references": [],
                            "utterance": "Also, can I dispute a parking ticket online?",
                        },
                        {
                            "turn_id": 6,
                            "role": "agent",
                            "da": "respond_solution",
                            "references": [
                                {"id_sp": "2", "doc_id": "Parking tickets#1_0", "label": "solution"}
                            ],
                            "utterance": "Yes, online with the ticket number and your plate",
                        },
                        {
                            "turn_id": 7,
                            "role": "agent",
                            "da": "respond_solution",
                            "references": [
                                {"id_sp": "9", "doc_id": "Unknown doc", "label": "solution"}
                            ],
                            "utterance": "This turn has no usable gold",
                        },
                    ],
                }
            ]
        }
    }
    (raw / evaluation.DOC_FILE).write_text(json.dumps(docs), encoding="utf-8")
    (raw / evaluation.DIAL_FILES["validation"]).write_text(json.dumps(dialogs), encoding="utf-8")
    return raw


def test_prepare_corpus_writes_verbatim_text_and_gold(raw_dir: Path, tmp_path: Path):
    corpus = tmp_path / "corpus"
    summary = evaluation.prepare_corpus(raw_dir, corpus)
    assert summary == {"documents": 2, "domains": {"dmv": 2}}
    gold = evaluation.load_gold(corpus)
    entry = gold["Insurance lapses#1_0"]
    assert entry["path"] == "dmv/insurance-lapses-1-0.txt"
    text = (corpus / entry["path"]).read_text(encoding="utf-8")
    assert text == INSURANCE_TEXT
    start, end = entry["spans"]["2"]
    assert text[start:end].startswith("You must get new insurance")


def test_load_turns_builds_history_and_gold_spans(raw_dir: Path, tmp_path: Path):
    corpus = tmp_path / "corpus"
    evaluation.prepare_corpus(raw_dir, corpus)
    turns = evaluation.load_turns(raw_dir, evaluation.load_gold(corpus))
    assert [t.turn_id for t in turns] == [2, 4, 6]
    second = turns[1]
    assert second.message == "How long do I have"
    assert second.history == [
        ("user", "My insurance ended so what should I do"),
        ("agent", "We will suspend your registration and license"),
    ]
    assert second.gold_spans[0][0] == "dmv/insurance-lapses-1-0.txt"
    assert second.reference_answer == "You have ten days to get new insurance"
    sample = evaluation.sample_turns(turns, 2, 7)
    assert len(sample) == 2 and sample == evaluation.sample_turns(turns, 2, 7)
    assert evaluation.sample_turns(turns, 10, 7) == turns


def make_hit(path: str, start: int, end: int) -> Hit:
    chunk = ChunkRecord(1, 1, "c", 0, start, end, "x" * (end - start), path, "t")
    return Hit(chunk=chunk, score=1.0)


def test_hit_matches_requires_half_overlap_of_the_shorter_span():
    gold = [("a.txt", 100, 200)]
    assert evaluation.hit_matches(make_hit("a.txt", 150, 400), gold)
    assert evaluation.hit_matches(make_hit("a.txt", 0, 160), gold)
    assert not evaluation.hit_matches(make_hit("a.txt", 0, 120), gold)
    assert not evaluation.hit_matches(make_hit("b.txt", 100, 200), gold)
    assert evaluation.hit_matches(make_hit("a.txt", 120, 130), gold)


def test_retrieval_run_metrics():
    turn = evaluation.EvalTurn("d", "dmv", 2, [], "q", [], "a", [("a.txt", 100, 200)])
    run = evaluation.RetrievalRun("window", "hybrid", "message", turns=2)
    run.record(turn, [make_hit("b.txt", 0, 50), make_hit("a.txt", 100, 200)], 12.0, False)
    run.record(turn, [make_hit("b.txt", 0, 50)], 8.0, True)
    row = run.to_dict()
    assert row["recall@1"] == 0.0 and row["recall@5"] == 0.5 and row["doc_recall@5"] == 0.5
    assert row["mrr@10"] == 0.25
    assert row["mean_ms"] == 10.0 and row["rewrite_fallbacks"] == 1
    assert row["recall@5_by_domain"] == {"dmv": 0.5}


def test_end_to_end_evaluation_with_fake_models(runtime, raw_dir: Path, tmp_path: Path):
    corpus = tmp_path / "md2d"
    evaluation.prepare_corpus(raw_dir, corpus)
    sync_folder(runtime.store, runtime.embedder, runtime.cfg, corpus, "md2d")
    turns = evaluation.load_turns(raw_dir, evaluation.load_gold(corpus))
    report = evaluation.evaluate_retrieval(runtime, turns, "md2d", subset_size=2)
    assert report["turns_full"] == 3 and report["turns_subset"] == 2
    rows = report["rows"]
    assert len(rows) == len(evaluation.retrieval_grid(2))
    full_rows = [r for r in rows if r["scope"] == "full"]
    assert {r["rerank"] for r in full_rows} == {"none"}
    window = next(r for r in full_rows if r["strategy"] == "window")
    assert window["recall@10"] == 1.0
    reranked = [r for r in rows if r["rerank"] in ("message", "query")]
    assert len(reranked) == len(evaluation.RERANK_STRATEGIES) + 1
    assert all(r["abstention"][0]["coverage"] == 1.0 for r in reranked)
    assert all(0.0 <= r["recall@1"] <= r["recall@5"] <= r["recall@10"] <= 1.0 for r in rows)
    samples = tmp_path / "samples.jsonl"
    generation = evaluation.evaluate_generation(runtime, turns, "md2d", samples)
    assert generation["turns"] == 3 and generation["llm"] == "fake"
    assert generation["grounded_rate"] == 1.0
    assert generation["citation_gold_hit_rate"] >= 2 / 3
    assert len(samples.read_text(encoding="utf-8").splitlines()) == 3
    timing = evaluation.measure_ingest(runtime, corpus, "md2d", tmp_path / "work")
    assert timing["full_sync"]["added"] == 2
    assert timing["no_change_sync"]["unchanged"] == 2 and timing["no_change_sync"]["seconds"] >= 0
    assert timing["one_file_changed_sync"]["updated"] == 1
    assert runtime.store.list_documents("md2d-timing") == []
    stats = evaluation.corpus_stats(runtime, "md2d")
    assert stats["documents"] == 2 and stats["chunks"] >= 2


def test_generation_resumes_and_tolerates_llm_errors(runtime, raw_dir: Path, tmp_path: Path):
    from convrag.llm import LLMError

    corpus = tmp_path / "md2d"
    evaluation.prepare_corpus(raw_dir, corpus)
    sync_folder(runtime.store, runtime.embedder, runtime.cfg, corpus, "md2d")
    turns = evaluation.load_turns(raw_dir, evaluation.load_gold(corpus))
    samples = tmp_path / "samples.jsonl"
    default = runtime.llm.responder

    def flaky(system, messages):
        if "Latest user message: Also, can I dispute" in messages[-1]["content"]:
            raise LLMError("timed out")
        return default(system, messages)

    runtime.llm.responder = flaky
    first = evaluation.evaluate_generation(runtime, turns, "md2d", samples)
    assert first["turns"] == 2 and first["errors"] == 1
    runtime.llm.responder = default
    calls_before = len(runtime.llm.calls)
    second = evaluation.evaluate_generation(runtime, turns, "md2d", samples)
    assert second["turns"] == 3 and second["errors"] == 0
    assert len(runtime.llm.calls) == calls_before + 1
    rows = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert sum(1 for r in rows if r.get("error")) == 1
    third = evaluation.evaluate_generation(runtime, turns, "md2d", samples)
    assert third == second
    assert len(runtime.llm.calls) == calls_before + 1
