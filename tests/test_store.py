import numpy as np

from convrag.store import Store


def make_store(tmp_path):
    return Store(tmp_path / "test.db")


def add_doc(store, path, title, texts, collection="c"):
    rows = [(i, i * 100, i * 100 + len(t), t, t) for i, t in enumerate(texts)]
    vectors = np.eye(max(2, len(texts)), dtype=np.float32)[: len(texts)]
    return store.replace_document(collection, path, title, f"hash-{path}", 10, rows, vectors)


def test_replace_document_and_sparse_search(tmp_path):
    store = make_store(tmp_path)
    add_doc(store, "a.txt", "Fees", ["renewal costs sixty dollars", "bring proof of identity"])
    add_doc(store, "b.txt", "Plates", ["turn in your plates after a lapse"])
    hits = store.sparse_search('"plates"', "c", 10)
    assert len(hits) == 1
    chunk = store.get_chunks([hits[0].chunk_id])[hits[0].chunk_id]
    assert chunk.path == "b.txt"
    assert chunk.title == "Plates"
    assert store.sparse_search('"renew"', "c", 10)
    assert store.sparse_search('"plates"', "other", 10) == []


def test_replace_keeps_single_copy_and_bumps_version(tmp_path):
    store = make_store(tmp_path)
    before = store.index_version()
    add_doc(store, "a.txt", "Fees", ["one sentence"])
    add_doc(store, "a.txt", "Fees v2", ["another sentence entirely"])
    assert store.index_version() == before + 2
    docs = store.list_documents("c")
    assert len(docs) == 1 and docs[0].title == "Fees v2"
    assert store.chunk_count("c") == 1
    assert store.sparse_search('"one"', "c", 10) == []
    assert store.sparse_search('"another"', "c", 10)


def test_delete_document_removes_everything(tmp_path):
    store = make_store(tmp_path)
    add_doc(store, "a.txt", "Fees", ["alpha beta", "gamma delta"])
    assert store.delete_document("c", "a.txt")
    assert not store.delete_document("c", "a.txt")
    assert store.chunk_count() == 0
    assert store.sparse_search('"alpha"', "c", 10) == []
    ids, matrix = store.load_vectors("c")
    assert ids.size == 0 and matrix.shape == (0, 0)


def test_load_vectors_matches_chunk_ids(tmp_path):
    store = make_store(tmp_path)
    add_doc(store, "a.txt", "Fees", ["alpha", "beta", "gamma"])
    ids, matrix = store.load_vectors("c")
    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float32
    records = store.get_chunks(list(ids))
    assert [records[i].position for i in ids] == [0, 1, 2]
    np.testing.assert_allclose(matrix, np.eye(3, dtype=np.float32))


def test_collections_summary(tmp_path):
    store = make_store(tmp_path)
    add_doc(store, "a.txt", "A", ["x y"], collection="one")
    add_doc(store, "b.txt", "B", ["x y", "z"], collection="one")
    add_doc(store, "c.txt", "C", ["q"], collection="two")
    assert store.collections() == [
        {"collection": "one", "documents": 2, "chunks": 3},
        {"collection": "two", "documents": 1, "chunks": 1},
    ]


def test_sessions_messages_and_traces(tmp_path):
    store = make_store(tmp_path)
    session = store.create_session("c")
    assert store.create_session("c", session) == session
    store.add_message(session, "user", "hello")
    store.add_message(session, "assistant", "hi", {"sources": [1]})
    store.add_message(session, "user", "again")
    messages = store.get_messages(session)
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[1].payload == {"sources": [1]}
    assert [m.content for m in store.get_messages(session, limit=2)] == ["hi", "again"]
    assert store.get_session("missing") is None
    for i in range(7):
        store.add_trace("chat", session, {"n": i}, keep=5)
    traces = store.recent_traces(10)
    assert [t["n"] for t in traces] == [6, 5, 4, 3, 2]
