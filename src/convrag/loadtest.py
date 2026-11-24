import asyncio
import random
import statistics
import time

import httpx


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    collection: str,
    queries: list[str],
    pending: list[int],
    latencies: list[float],
    errors: list[str],
    rng: random.Random,
) -> None:
    while pending:
        pending.pop()
        query = rng.choice(queries)
        begin = time.perf_counter()
        try:
            response = await client.post(
                url, json={"query": query, "collection": collection, "top_k": 5}
            )
            if response.status_code != 200:
                errors.append(f"status {response.status_code}")
        except httpx.HTTPError as exc:
            errors.append(str(exc))
        latencies.append((time.perf_counter() - begin) * 1000)


async def _run(
    base_url: str, collection: str, queries: list[str], requests: int, concurrency: int, seed: int
) -> dict:
    url = base_url.rstrip("/") + "/search"
    pending = list(range(requests))
    latencies: list[float] = []
    errors: list[str] = []
    rng = random.Random(seed)
    async with httpx.AsyncClient(timeout=60.0) as client:
        warm = await client.get(base_url.rstrip("/") + "/ready")
        warm.raise_for_status()
        begin = time.perf_counter()
        await asyncio.gather(
            *[
                _worker(client, url, collection, queries, pending, latencies, errors, rng)
                for _ in range(concurrency)
            ]
        )
        elapsed = time.perf_counter() - begin
    return {
        "endpoint": "/search",
        "requests": requests,
        "concurrency": concurrency,
        "seconds": round(elapsed, 2),
        "requests_per_second": round(requests / elapsed, 1) if elapsed else 0.0,
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "errors": len(errors),
        "error_samples": errors[:5],
    }


def run_loadtest(
    base_url: str,
    collection: str,
    queries: list[str],
    requests: int = 500,
    concurrency: int = 8,
    seed: int = 7,
) -> dict:
    if not queries:
        raise ValueError("load test needs at least one query")
    return asyncio.run(_run(base_url, collection, queries, requests, concurrency, seed))
