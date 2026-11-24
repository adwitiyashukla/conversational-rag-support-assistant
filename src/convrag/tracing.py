import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

BUCKETS_MS = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0)


@dataclass
class Span:
    name: str
    ms: float
    meta: dict = field(default_factory=dict)


@dataclass
class Trace:
    kind: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @contextmanager
    def span(self, name: str, **meta) -> Iterator[Span]:
        begin = time.perf_counter()
        record = Span(name=name, ms=0.0, meta=dict(meta))
        try:
            yield record
        finally:
            record.ms = round((time.perf_counter() - begin) * 1000, 2)
            self.spans.append(record)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.id,
            "kind": self.kind,
            "total_ms": self.total_ms,
            "spans": [{"name": s.name, "ms": s.ms, **s.meta} for s in self.spans],
            **self.meta,
        }


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list] = {}

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self.lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe_ms(self, name: str, ms: float, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self.lock:
            entry = self.histograms.get(key)
            if entry is None:
                entry = [[0] * len(BUCKETS_MS), 0.0, 0]
                self.histograms[key] = entry
            for index, bound in enumerate(BUCKETS_MS):
                if ms <= bound:
                    entry[0][index] += 1
            entry[1] += ms
            entry[2] += 1

    def record_trace(self, trace: Trace) -> None:
        self.inc("convrag_requests_total", kind=trace.kind)
        self.observe_ms("convrag_request_ms", trace.total_ms, kind=trace.kind)
        for span in trace.spans:
            self.observe_ms("convrag_stage_ms", span.ms, kind=trace.kind, stage=span.name)

    @staticmethod
    def _labels(labels: tuple[tuple[str, str], ...], extra: str = "") -> str:
        parts = [f'{k}="{v}"' for k, v in labels]
        if extra:
            parts.append(extra)
        return "{" + ",".join(parts) + "}" if parts else ""

    def render(self) -> str:
        lines = []
        with self.lock:
            names = sorted({k[0] for k in self.counters})
            for name in names:
                lines.append(f"# TYPE {name} counter")
                for (metric, labels), value in sorted(self.counters.items()):
                    if metric == name:
                        lines.append(f"{name}{self._labels(labels)} {value:g}")
            names = sorted({k[0] for k in self.histograms})
            for name in names:
                lines.append(f"# TYPE {name} histogram")
                for (metric, labels), (buckets, total, count) in sorted(self.histograms.items()):
                    if metric != name:
                        continue
                    for bound, cumulative in zip(BUCKETS_MS, buckets, strict=True):
                        le = f'le="{bound:g}"'
                        lines.append(f"{name}_bucket{self._labels(labels, le)} {cumulative}")
                    inf = 'le="+Inf"'
                    lines.append(f"{name}_bucket{self._labels(labels, inf)} {count}")
                    lines.append(f"{name}_sum{self._labels(labels)} {total:g}")
                    lines.append(f"{name}_count{self._labels(labels)} {count}")
        return "\n".join(lines) + "\n"
