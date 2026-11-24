import numpy as np

from convrag.embeddings import HashEmbedder, normalize_rows
from convrag.retrieval import Retriever, fts_match, rrf
from convrag.tracing import Trace


def test_fts_match_drops_stopwords_and_quotes_tokens():
    assert fts_match("What is the fee for plates?", 64) == '"fee" OR "plates"'
    assert fts_match("the of and", 64) == ""
    assert fts_match("one two three four", 2) == '"three" OR "four"'
    assert fts_match("don't pay", 64) == '"dont" OR "pay"'


def test_rrf_rewards_items_present_in_several_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert fused[1] > fused[3] > fused[2] > fused[4]
    assert fused[1] == 1 / 61 + 1 / 62


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(64)
    a = embedder.embed_documents(["insurance lapse letter", "insurance lapse letter"])
    np.testing.assert_allclose(a[0], a[1])
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    b = embedder.embed_queries(["parking ticket dispute"])[0]
    assert float(a[0] @ b) < float(a[0] @ embedder.embed_queries(["insurance lapse"])[0])
    assert normalize_rows(np.zeros((1, 4), dtype=np.float32)).sum() == 0


def test_bm25_dense_and_hybrid_find_the_right_document(runtime):
    retriever: Retriever = runtime.retriever
    for mode in ("bm25", "dense", "hybrid"):
        hits = retriever.search(["my insurance ended what happens"], "dmv", mode=mode, rerank=False)
        assert hits, mode
        assert hits[0].chunk.path == "dmv/insurance.txt", mode
    hits = retriever.search(["dispute a parking ticket"], "dmv", mode="hybrid", rerank=False)
    assert hits[0].chunk.path == "parking.html"
    assert hits[0].sparse_rank is not None and hits[0].dense_rank is not None


def test_rerank_orders_by_rerank_score_and_records_trace(runtime):
    trace = Trace(kind="search")
    hits = runtime.retriever.search(["renew license online"], "dmv", top_k=3, trace=trace)
    assert hits[0].chunk.path == "dmv/licenses.md"
    scores = [h.rerank_score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert {s.name for s in trace.spans} == {"sparse", "dense", "fuse", "rerank"}
    payload = hits[0].to_dict(max_chars=20)
    assert payload["text"].endswith("...") and len(payload["text"]) <= 23


def test_fusion_over_two_queries_marks_query_sources(runtime):
    hits = runtime.retriever.search(
        ["parking ticket", "insurance suspend registration"], "dmv", mode="bm25", rerank=False
    )
    paths = {h.chunk.path for h in hits}
    assert {"parking.html", "dmv/insurance.txt"} <= paths
    assert all(h.queries for h in hits)


def test_empty_queries_and_unknown_collection(runtime):
    assert runtime.retriever.search(["", "   "], "dmv") == []
    assert runtime.retriever.search(["insurance"], "nothing-here") == []


def test_vector_index_reloads_after_ingest(runtime, cfg, corpus):
    from convrag.ingest import sync_folder

    before = len(runtime.retriever.vectors.get("dmv")[0])
    (corpus / "boats.txt").write_text("Boat registration\n\nRegister your boat yearly.", "utf-8")
    sync_folder(runtime.store, runtime.embedder, cfg, corpus, "dmv")
    after = len(runtime.retriever.vectors.get("dmv")[0])
    assert after == before + 1
    hits = runtime.retriever.search(["register my boat"], "dmv", mode="dense", rerank=False)
    assert hits[0].chunk.path == "boats.txt"


def test_rerank_follows_config_unless_requested(runtime):
    runtime.cfg.reranker.enabled = False
    hits = runtime.retriever.search(["renew license online"], "dmv")
    assert hits and all(h.rerank_score is None for h in hits)
    hits = runtime.retriever.search(["renew license online"], "dmv", rerank=True)
    assert hits and all(h.rerank_score is not None for h in hits)
    runtime.cfg.reranker.enabled = True
    hits = runtime.retriever.search(["renew license online"], "dmv")
    assert hits and all(h.rerank_score is not None for h in hits)
