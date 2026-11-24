import json
import random
import re
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from convrag.chat import ChatResult
from convrag.config import Config
from convrag.ingest import sync_folder
from convrag.llm import LLMError
from convrag.retrieval import Hit
from convrag.rewrite import STRATEGIES, Formulation, formulate
from convrag.runtime import Runtime
from convrag.text import exact_match, token_f1

DOC_FILE = "multidoc2dial_doc.json"
DIAL_FILES = {
    "train": "multidoc2dial_dial_train.json",
    "validation": "multidoc2dial_dial_validation.json",
}
GOLD_FILE = "gold.json"
KS = (1, 5, 10)
THRESHOLDS = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0)
RERANK_STRATEGIES = ("last_turn", "window", "rewrite")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:120] or "doc"


def prepare_corpus(raw_dir: Path, corpus_dir: Path) -> dict:
    source = raw_dir / DOC_FILE
    if not source.exists():
        raise FileNotFoundError(f"missing {source}")
    data = json.loads(source.read_text(encoding="utf-8"))["doc_data"]
    corpus_dir.mkdir(parents=True, exist_ok=True)
    gold: dict[str, dict] = {}
    used: set[str] = set()
    counts: dict[str, int] = {}
    for domain, docs in data.items():
        folder = corpus_dir / domain
        folder.mkdir(parents=True, exist_ok=True)
        for doc_id, doc in docs.items():
            slug = slugify(doc_id)
            candidate = slug
            suffix = 2
            while candidate in used:
                candidate = f"{slug}-{suffix}"
                suffix += 1
            used.add(candidate)
            rel = f"{domain}/{candidate}.txt"
            with (corpus_dir / rel).open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(doc["doc_text"])
            gold[doc_id] = {
                "path": rel,
                "title": doc["title"],
                "domain": domain,
                "spans": {
                    sp_id: [span["start_sp"], span["end_sp"]]
                    for sp_id, span in doc["spans"].items()
                },
            }
            counts[domain] = counts.get(domain, 0) + 1
    (corpus_dir / GOLD_FILE).write_text(json.dumps({"documents": gold}), encoding="utf-8")
    return {"documents": len(gold), "domains": counts}


def load_gold(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / GOLD_FILE).read_text(encoding="utf-8"))["documents"]


@dataclass
class EvalTurn:
    dial_id: str
    domain: str
    turn_id: int
    history: list[tuple[str, str]]
    message: str
    references: list[tuple[str, str]]
    reference_answer: str
    gold_spans: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.dial_id}:{self.turn_id}"


def load_turns(raw_dir: Path, gold: dict, split: str = "validation") -> list[EvalTurn]:
    source = raw_dir / DIAL_FILES[split]
    data = json.loads(source.read_text(encoding="utf-8"))["dial_data"]
    turns: list[EvalTurn] = []
    for domain, dialogs in data.items():
        for dialog in dialogs:
            ordered = sorted(dialog["turns"], key=lambda t: t["turn_id"])
            for index, turn in enumerate(ordered):
                if turn["role"] != "agent" or not turn["references"] or index == 0:
                    continue
                previous = ordered[index - 1]
                if previous["role"] != "user":
                    continue
                history = [(t["role"], t["utterance"]) for t in ordered[: index - 1]]
                references = [(r["doc_id"], r["id_sp"]) for r in turn["references"]]
                spans = []
                for doc_id, sp_id in references:
                    doc = gold.get(doc_id)
                    if doc and sp_id in doc["spans"]:
                        start, end = doc["spans"][sp_id]
                        spans.append((doc["path"], start, end))
                if not spans:
                    continue
                turns.append(
                    EvalTurn(
                        dial_id=dialog["dial_id"],
                        domain=domain,
                        turn_id=turn["turn_id"],
                        history=history,
                        message=previous["utterance"],
                        references=references,
                        reference_answer=turn["utterance"],
                        gold_spans=spans,
                    )
                )
    return turns


def sample_turns(turns: list[EvalTurn], size: int, seed: int) -> list[EvalTurn]:
    if size >= len(turns):
        return list(turns)
    rng = random.Random(seed)
    return sorted(rng.sample(turns, size), key=lambda t: (t.dial_id, t.turn_id))


def hit_matches(hit: Hit, gold_spans: list[tuple[str, int, int]]) -> bool:
    for path, start, end in gold_spans:
        if hit.chunk.path != path:
            continue
        overlap = min(hit.chunk.end, end) - max(hit.chunk.start, start)
        shortest = max(1, min(end - start, hit.chunk.end - hit.chunk.start))
        if overlap > 0 and overlap / shortest >= 0.5:
            return True
    return False


def gold_paths(gold_spans: list[tuple[str, int, int]]) -> set[str]:
    return {path for path, _, _ in gold_spans}


@dataclass
class RetrievalRun:
    strategy: str
    mode: str
    rerank: str
    turns: int
    passage_hits: dict[int, int] = field(default_factory=lambda: {k: 0 for k in KS})
    document_hits: dict[int, int] = field(default_factory=lambda: {k: 0 for k in KS})
    reciprocal_ranks: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    fallbacks: int = 0
    per_domain: dict[str, list[int]] = field(default_factory=dict)
    top_scores: list[tuple[float, bool]] = field(default_factory=list)

    def record(self, turn: EvalTurn, hits: list[Hit], ms: float, fallback: bool) -> None:
        first_rank = None
        paths = gold_paths(turn.gold_spans)
        first_doc_rank = None
        for rank, hit in enumerate(hits, start=1):
            if first_rank is None and hit_matches(hit, turn.gold_spans):
                first_rank = rank
            if first_doc_rank is None and hit.chunk.path in paths:
                first_doc_rank = rank
        for k in KS:
            if first_rank is not None and first_rank <= k:
                self.passage_hits[k] += 1
            if first_doc_rank is not None and first_doc_rank <= k:
                self.document_hits[k] += 1
        self.reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        self.latencies_ms.append(ms)
        self.fallbacks += int(fallback)
        hit_at_5 = first_rank is not None and first_rank <= 5
        bucket = self.per_domain.setdefault(turn.domain, [0, 0])
        bucket[0] += int(hit_at_5)
        bucket[1] += 1
        if hits and hits[0].rerank_score is not None:
            self.top_scores.append((hits[0].rerank_score, hit_at_5))

    def to_dict(self) -> dict:
        n = max(1, self.turns)
        return {
            "strategy": self.strategy,
            "mode": self.mode,
            "rerank": self.rerank,
            "turns": self.turns,
            **{f"recall@{k}": round(self.passage_hits[k] / n, 4) for k in KS},
            **{f"doc_recall@{k}": round(self.document_hits[k] / n, 4) for k in KS},
            "mrr@10": round(sum(self.reciprocal_ranks) / n, 4),
            "mean_ms": round(statistics.fmean(self.latencies_ms), 1) if self.latencies_ms else 0,
            "p95_ms": round(_percentile(self.latencies_ms, 95), 1) if self.latencies_ms else 0,
            "rewrite_fallbacks": self.fallbacks,
            "recall@5_by_domain": {
                d: round(v[0] / max(1, v[1]), 4) for d, v in sorted(self.per_domain.items())
            },
            "abstention": abstention_curve(self.top_scores, n),
        }


def abstention_curve(top_scores: list[tuple[float, bool]], total: int) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        answered = [hit for score, hit in top_scores if score >= threshold]
        rows.append(
            {
                "min_score": threshold,
                "coverage": round(len(answered) / max(1, total), 4),
                "recall@5_when_answered": round(sum(answered) / max(1, len(answered)), 4),
            }
        )
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


class FormulationCache:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.cache: dict[tuple[str, str], Formulation] = {}
        self.llm_ms: list[float] = []

    def get(self, turn: EvalTurn, strategy: str) -> Formulation:
        key = (strategy, turn.key)
        if key not in self.cache:
            begin = time.perf_counter()
            self.cache[key] = formulate(
                turn.history, turn.message, self.runtime.cfg.rewrite, self.runtime.llm, strategy
            )
            if strategy == "rewrite":
                self.llm_ms.append((time.perf_counter() - begin) * 1000)
        return self.cache[key]


def run_retrieval(
    runtime: Runtime,
    turns: list[EvalTurn],
    collection: str,
    strategy: str,
    mode: str,
    rerank: str,
    cache: FormulationCache,
    progress=None,
) -> RetrievalRun:
    run = RetrievalRun(strategy=strategy, mode=mode, rerank=rerank, turns=len(turns))
    top_k = max(KS)
    for index, turn in enumerate(turns, start=1):
        formulation = cache.get(turn, strategy)
        begin = time.perf_counter()
        hits = runtime.retriever.search(
            formulation.queries,
            collection,
            mode=mode,
            top_k=top_k,
            rerank=rerank != "none",
            rerank_query=turn.message if rerank == "message" else formulation.rerank_query,
        )
        ms = (time.perf_counter() - begin) * 1000
        run.record(turn, hits, ms, formulation.fallback)
        if progress and index % 250 == 0:
            progress(f"  {strategy} {mode} rerank={rerank}: {index}/{len(turns)}")
    return run


def retrieval_grid(subset_size: int) -> list[tuple[str, str, str, str]]:
    grid = []
    for strategy in ("last_turn", "window", "history", "fusion"):
        grid.append(("full", strategy, "hybrid", "none"))
    if subset_size > 0:
        for strategy in STRATEGIES:
            grid.append(("subset", strategy, "hybrid", "none"))
        for mode in ("bm25", "dense"):
            grid.append(("subset", "window", mode, "none"))
        for strategy in RERANK_STRATEGIES:
            grid.append(("subset", strategy, "hybrid", "message"))
        grid.append(("subset", "window", "hybrid", "query"))
    return grid


def evaluate_retrieval(
    runtime: Runtime, turns: list[EvalTurn], collection: str, subset_size: int, progress=None
) -> dict:
    cache = FormulationCache(runtime)
    subset = sample_turns(turns, subset_size, runtime.cfg.evaluation.seed) if subset_size else []
    rows = []
    for scope, strategy, mode, rerank in retrieval_grid(subset_size):
        chosen = turns if scope == "full" else subset
        if progress:
            progress(f"retrieval {scope}: strategy={strategy} mode={mode} rerank={rerank}")
        run = run_retrieval(runtime, chosen, collection, strategy, mode, rerank, cache, progress)
        rows.append({"scope": scope, **run.to_dict()})
    return {
        "turns_full": len(turns),
        "turns_subset": len(subset),
        "rewrite_llm_mean_ms": round(statistics.fmean(cache.llm_ms), 1) if cache.llm_ms else None,
        "rows": rows,
    }


def _load_samples(samples_path: Path) -> dict[str, dict]:
    if not samples_path.exists():
        return {}
    rows = {}
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["key"]] = row
    return rows


def generation_row(turn: EvalTurn, result: ChatResult) -> dict:
    cited = [result.hits[i - 1] for i in result.verified.cited_sources if i <= len(result.hits)]
    return {
        "key": turn.key,
        "domain": turn.domain,
        "message": turn.message,
        "queries": result.formulation.queries,
        "answer": result.answer,
        "raw_answer": result.raw_answer,
        "reference": turn.reference_answer,
        "f1": round(token_f1(result.answer, turn.reference_answer), 4),
        "em": exact_match(result.answer, turn.reference_answer),
        "abstained": result.abstained,
        "grounded": result.grounded,
        "cited_sources": result.verified.cited_sources,
        "cited_gold_hit": any(hit_matches(h, turn.gold_spans) for h in cited) if cited else None,
        "gold_hit_in_top_k": any(hit_matches(h, turn.gold_spans) for h in result.hits),
        "total_ms": result.trace.total_ms,
        "stages": {span.name: span.ms for span in result.trace.spans},
        "error": None,
    }


def summarize_generation(rows: list[dict], llm: str, strategy: str) -> dict:
    done = [r for r in rows if not r.get("error")]
    n = max(1, len(done))
    abstained = sum(1 for r in done if r["abstained"])
    cited = [r for r in done if r["cited_gold_hit"] is not None]
    stage_ms: dict[str, list[float]] = {}
    for r in done:
        for name, ms in r["stages"].items():
            stage_ms.setdefault(name, []).append(ms)
    latencies = [r["total_ms"] for r in done]
    return {
        "turns": len(done),
        "errors": len(rows) - len(done),
        "llm": llm,
        "strategy": strategy,
        "f1": round(statistics.fmean(r["f1"] for r in done), 4) if done else 0.0,
        "em": round(statistics.fmean(r["em"] for r in done), 4) if done else 0.0,
        "abstain_rate": round(abstained / n, 4),
        "grounded_rate": round(sum(1 for r in done if r["grounded"]) / n, 4),
        "answered_grounded_rate": round(
            sum(1 for r in done if r["grounded"]) / max(1, len(done) - abstained), 4
        ),
        "gold_in_top_k_rate": round(sum(1 for r in done if r["gold_hit_in_top_k"]) / n, 4),
        "citation_gold_hit_rate": round(
            sum(1 for r in cited if r["cited_gold_hit"]) / max(1, len(cited)), 4
        ),
        "answers_with_citations": len(cited),
        "mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "p95_ms": round(_percentile(latencies, 95), 1),
        "stage_mean_ms": {k: round(statistics.fmean(v), 1) for k, v in sorted(stage_ms.items())},
    }


def evaluate_generation(
    runtime: Runtime, turns: list[EvalTurn], collection: str, samples_path: Path, progress=None
) -> dict:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_samples(samples_path)
    rows: list[dict] = []
    with samples_path.open("a", encoding="utf-8") as handle:
        for index, turn in enumerate(turns, start=1):
            previous = existing.get(turn.key)
            if previous and not previous.get("error"):
                rows.append(previous)
                continue
            try:
                row = generation_row(
                    turn, runtime.chat.answer(turn.history, turn.message, collection)
                )
            except LLMError as exc:
                row = {"key": turn.key, "domain": turn.domain, "error": str(exc)}
                if progress:
                    progress(f"  generation: turn {turn.key} failed: {exc}")
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if progress and index % 25 == 0:
                done = [r for r in rows if not r.get("error")]
                mean_f1 = statistics.fmean(r["f1"] for r in done) if done else 0.0
                progress(f"  generation: {index}/{len(turns)} mean f1 {mean_f1:.3f}")
    return summarize_generation(rows, runtime.llm.name, runtime.cfg.rewrite.strategy)


def measure_ingest(runtime: Runtime, corpus_dir: Path, collection: str, work_dir: Path) -> dict:
    scratch = work_dir / "ingest-timing"
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(corpus_dir, scratch, ignore=shutil.ignore_patterns("*.json"))
    temp_collection = f"{collection}-timing"
    first = sync_folder(runtime.store, runtime.embedder, runtime.cfg, scratch, temp_collection)
    second = sync_folder(runtime.store, runtime.embedder, runtime.cfg, scratch, temp_collection)
    target = sorted(p for p in scratch.rglob("*.txt"))[0]
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\nThis line was appended to measure incremental ingestion.\n")
    third = sync_folder(runtime.store, runtime.embedder, runtime.cfg, scratch, temp_collection)
    for document in runtime.store.list_documents(temp_collection):
        runtime.store.delete_document(temp_collection, document.path)
    shutil.rmtree(scratch)
    return {
        "full_sync": first.to_dict(),
        "no_change_sync": second.to_dict(),
        "one_file_changed_sync": third.to_dict(),
    }


def write_results(results_dir: Path, name: str, payload: dict) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def corpus_stats(runtime: Runtime, collection: str) -> dict:
    documents = runtime.store.list_documents(collection)
    return {
        "collection": collection,
        "documents": len(documents),
        "chunks": runtime.store.chunk_count(collection),
        "embedding_model": runtime.embedder.name,
        "reranker": runtime.reranker.name if runtime.reranker else None,
    }


def config_snapshot(cfg: Config) -> dict:
    return {
        "chunking": vars(cfg.chunking),
        "retrieval": vars(cfg.retrieval),
        "reranker": {k: v for k, v in vars(cfg.reranker).items()},
        "rewrite": vars(cfg.rewrite),
        "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model},
        "generation": vars(cfg.generation),
    }
