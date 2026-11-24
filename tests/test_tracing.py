from convrag.tracing import BUCKETS_MS, Metrics, Trace


def test_trace_records_spans_in_order():
    trace = Trace(kind="chat")
    with trace.span("first", items=3):
        pass
    with trace.span("second") as span:
        span.meta["chars"] = 10
    payload = trace.to_dict()
    assert [s["name"] for s in payload["spans"]] == ["first", "second"]
    assert payload["spans"][0]["items"] == 3 and payload["spans"][1]["chars"] == 10
    assert payload["total_ms"] >= sum(s["ms"] for s in payload["spans"]) * 0.5
    assert payload["kind"] == "chat" and len(payload["trace_id"]) == 12


def test_metrics_render_prometheus_text():
    metrics = Metrics()
    metrics.inc("convrag_requests_total", kind="chat")
    metrics.inc("convrag_requests_total", kind="chat")
    metrics.observe_ms("convrag_stage_ms", 7.0, stage="dense")
    metrics.observe_ms("convrag_stage_ms", 70.0, stage="dense")
    text = metrics.render()
    assert "# TYPE convrag_requests_total counter" in text
    assert 'convrag_requests_total{kind="chat"} 2' in text
    assert 'convrag_stage_ms_bucket{stage="dense",le="5"} 0' in text
    assert 'convrag_stage_ms_bucket{stage="dense",le="10"} 1' in text
    assert 'convrag_stage_ms_bucket{stage="dense",le="100"} 2' in text
    assert 'convrag_stage_ms_bucket{stage="dense",le="+Inf"} 2' in text
    assert 'convrag_stage_ms_sum{stage="dense"} 77' in text
    assert 'convrag_stage_ms_count{stage="dense"} 2' in text
    assert text.count("_bucket{") == len(BUCKETS_MS) + 1


def test_record_trace_feeds_counters_and_histograms():
    metrics = Metrics()
    trace = Trace(kind="search")
    with trace.span("sparse"):
        pass
    metrics.record_trace(trace)
    text = metrics.render()
    assert 'convrag_requests_total{kind="search"} 1' in text
    assert 'convrag_stage_ms_count{kind="search",stage="sparse"} 1' in text
    assert 'convrag_request_ms_count{kind="search"} 1' in text
