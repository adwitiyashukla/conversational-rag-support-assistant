import argparse
import json
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from convrag import evaluation
from convrag.config import Config, ConfigError, load
from convrag.ingest import sync_folder
from convrag.llm import LLMError
from convrag.runtime import Runtime

DATA_URL = "http://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def cmd_data(cfg: Config, args: argparse.Namespace) -> int:
    raw_dir = cfg.resolve(cfg.evaluation.raw_dir)
    if not (raw_dir / evaluation.DOC_FILE).exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        archive = raw_dir.parent / "multidoc2dial.zip"
        log(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                name = Path(member).name
                if member.startswith("multidoc2dial/") and name and "__MACOSX" not in member:
                    (raw_dir / name).write_bytes(bundle.read(member))
        archive.unlink()
    corpus_dir = cfg.resolve(cfg.evaluation.corpus_dir)
    summary = evaluation.prepare_corpus(raw_dir, corpus_dir)
    log(f"corpus ready at {corpus_dir}: {summary}")
    return 0


def cmd_ingest(cfg: Config, args: argparse.Namespace) -> int:
    runtime = Runtime.build(cfg)
    try:
        report = sync_folder(
            runtime.store, runtime.embedder, cfg, Path(args.folder).resolve(), args.collection
        )
    finally:
        runtime.close()
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_search(cfg: Config, args: argparse.Namespace) -> int:
    runtime = Runtime.build(cfg)
    try:
        hits = runtime.retriever.search(
            [args.query],
            args.collection,
            mode=args.mode,
            top_k=args.top_k,
            rerank=not args.no_rerank,
        )
    finally:
        runtime.close()
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2))
        return 0
    for index, hit in enumerate(hits, start=1):
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        print(f"[{index}] {hit.chunk.title} ({hit.chunk.path}) score={score:.4f}")
        print("    " + hit.chunk.text.replace("\n", " ")[:300])
    return 0


def cmd_chat(cfg: Config, args: argparse.Namespace) -> int:
    runtime = Runtime.build(cfg)
    session_id = runtime.store.create_session(args.collection, args.session)
    print(f"session {session_id}, collection {args.collection}, type quit to exit")
    try:
        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message or message.lower() in ("quit", "exit"):
                break
            try:
                for event in runtime.chat.respond_stream(message, args.collection, session_id):
                    if event["type"] == "queries":
                        print(f"  searched: {' | '.join(event['queries'])}")
                    elif event["type"] == "token":
                        print(event["text"], end="", flush=True)
                    elif event["type"] == "done":
                        print()
                        if event["abstained"] or not event["grounded"]:
                            print(f"convrag> {event['answer']}")
                        for i, source in enumerate(event["sources"], start=1):
                            mark = "*" if i in event["cited_sources"] else " "
                            print(f"  {mark}[{i}] {source['title']} ({source['path']})")
                        print(f"  {event['trace']['total_ms']} ms")
            except LLMError as exc:
                print(f"llm error: {exc}")
    finally:
        runtime.close()
    return 0


def cmd_serve(cfg: Config, args: argparse.Namespace) -> int:
    import uvicorn

    from convrag.api import create_app

    app = create_app(cfg)
    uvicorn.run(app, host=args.host or cfg.server.host, port=args.port or cfg.server.port)
    return 0


def cmd_stats(cfg: Config, args: argparse.Namespace) -> int:
    runtime = Runtime.build(cfg)
    try:
        print(json.dumps(runtime.store.collections(), indent=2))
    finally:
        runtime.close()
    return 0


def _load_turns(cfg: Config, limit: int | None) -> list[evaluation.EvalTurn]:
    raw_dir = cfg.resolve(cfg.evaluation.raw_dir)
    corpus_dir = cfg.resolve(cfg.evaluation.corpus_dir)
    gold = evaluation.load_gold(corpus_dir)
    turns = evaluation.load_turns(raw_dir, gold, "validation")
    if limit:
        turns = evaluation.sample_turns(turns, limit, cfg.evaluation.seed)
    return turns


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    runtime = Runtime.build(cfg)
    results_dir = cfg.resolve(cfg.evaluation.results_dir)
    collection = cfg.evaluation.collection
    corpus_dir = cfg.resolve(cfg.evaluation.corpus_dir)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        if runtime.store.chunk_count(collection) == 0 and args.what != "ingest":
            log(
                f"collection {collection} is empty, run: convrag ingest {corpus_dir} "
                f"--collection {collection}"
            )
            return 1
        base = {
            "created_at": stamp,
            "corpus": evaluation.corpus_stats(runtime, collection),
            "config": evaluation.config_snapshot(cfg),
        }
        if args.what in ("retrieval", "all"):
            turns = _load_turns(cfg, args.limit)
            log(f"retrieval evaluation over {len(turns)} agent turns")
            began = time.perf_counter()
            payload = evaluation.evaluate_retrieval(
                runtime, turns, collection, args.subset, progress=log
            )
            payload["seconds"] = round(time.perf_counter() - began, 1)
            path = evaluation.write_results(results_dir, "retrieval", {**base, **payload})
            _print_retrieval(payload["rows"])
            log(f"wrote {path}")
        if args.what in ("generation", "all"):
            turns = evaluation.sample_turns(
                _load_turns(cfg, None),
                args.sample or cfg.evaluation.generation_sample,
                cfg.evaluation.seed,
            )
            log(f"generation evaluation over {len(turns)} turns with {runtime.llm.name}")
            began = time.perf_counter()
            payload = evaluation.evaluate_generation(
                runtime, turns, collection, results_dir / "generation_samples.jsonl", progress=log
            )
            payload["seconds"] = round(time.perf_counter() - began, 1)
            path = evaluation.write_results(results_dir, "generation", {**base, **payload})
            print(json.dumps(payload, indent=2))
            log(f"wrote {path}")
        if args.what in ("ingest", "all"):
            log("measuring full, unchanged and one-file-changed ingestion")
            payload = evaluation.measure_ingest(runtime, corpus_dir, collection, results_dir)
            path = evaluation.write_results(
                results_dir, "ingest", {"created_at": stamp, "config": base["config"], **payload}
            )
            print(json.dumps(payload, indent=2))
            log(f"wrote {path}")
    finally:
        runtime.close()
    return 0


def _print_retrieval(rows: list[dict]) -> None:
    columns = ["scope", "strategy", "mode", "rerank", "r@1", "r@5", "r@10", "doc@5", "mrr", "ms"]
    widths = [7, 10, 7, 7, 6, 6, 6, 6, 6, 7]
    print(
        " ".join(
            name.rjust(w) if i > 3 else name.ljust(w)
            for i, (name, w) in enumerate(zip(columns, widths, strict=True))
        )
    )
    for row in rows:
        print(
            f"{row['scope']:7} {row['strategy']:10} {row['mode']:7} {row['rerank']:7} "
            f"{row['recall@1']:6.3f} {row['recall@5']:6.3f} {row['recall@10']:6.3f} "
            f"{row['doc_recall@5']:6.3f} {row['mrr@10']:6.3f} {row['mean_ms']:7.1f}"
        )


def cmd_loadtest(cfg: Config, args: argparse.Namespace) -> int:
    from convrag.loadtest import run_loadtest

    turns = _load_turns(cfg, args.queries)
    queries = [t.message for t in turns] or ["how do I renew my license"]
    result = run_loadtest(
        args.url,
        cfg.evaluation.collection,
        queries,
        args.requests,
        args.concurrency,
        cfg.evaluation.seed,
    )
    results_dir = cfg.resolve(cfg.evaluation.results_dir)
    path = evaluation.write_results(
        results_dir,
        "loadtest",
        {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **result},
    )
    print(json.dumps(result, indent=2))
    log(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="convrag")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data")
    data.add_argument("action", choices=["prepare"])
    data.set_defaults(func=cmd_data)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("folder")
    ingest.add_argument("--collection", required=True)
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--collection", required=True)
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid"])
    search.add_argument("--top-k", type=int)
    search.add_argument("--no-rerank", action="store_true")
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=cmd_search)

    chat = sub.add_parser("chat")
    chat.add_argument("--collection", required=True)
    chat.add_argument("--session")
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=cmd_serve)

    stats = sub.add_parser("stats")
    stats.set_defaults(func=cmd_stats)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("what", choices=["retrieval", "generation", "ingest", "all"])
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--subset", type=int, default=500)
    evaluate.add_argument("--sample", type=int)
    evaluate.set_defaults(func=cmd_eval)

    loadtest = sub.add_parser("loadtest")
    loadtest.add_argument("--url", default="http://127.0.0.1:8000")
    loadtest.add_argument("--requests", type=int, default=500)
    loadtest.add_argument("--concurrency", type=int, default=8)
    loadtest.add_argument("--queries", type=int, default=200)
    loadtest.set_defaults(func=cmd_loadtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load(args.config)
    except ConfigError as exc:
        log(f"config error: {exc}")
        return 2
    try:
        return args.func(cfg, args)
    except (FileNotFoundError, ValueError) as exc:
        log(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
