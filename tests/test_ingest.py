from pathlib import Path

from convrag.embeddings import HashEmbedder
from convrag.ingest import load_document, scan_folder, sync_folder
from convrag.store import Store


def test_loaders_extract_titles_and_text(corpus: Path):
    md = load_document(corpus / "dmv" / "licenses.md")
    assert md.title == "Renewing a driver license"
    assert "vision test" in md.text
    txt = load_document(corpus / "dmv" / "insurance.txt")
    assert txt.title == "Insurance lapses"
    html = load_document(corpus / "parking.html")
    assert html.title == "Parking tickets"
    assert "var x" not in html.text
    assert "dispute a parking ticket" in html.text


def test_scan_folder_filters_by_extension_and_size(corpus: Path):
    files = scan_folder(corpus, [".md", ".txt", ".html"], 25)
    assert [f.name for f in files] == ["insurance.txt", "licenses.md", "parking.html"]
    assert scan_folder(corpus, [".csv"], 0) == []


def test_sync_is_incremental(cfg, corpus: Path):
    store = Store(cfg.resolve(cfg.paths.db_file))
    embedder = HashEmbedder(64)
    first = sync_folder(store, embedder, cfg, corpus, "dmv")
    assert (first.added, first.updated, first.unchanged, first.removed) == (3, 0, 0, 0)
    assert first.chunks_embedded == store.chunk_count("dmv") > 0
    second = sync_folder(store, embedder, cfg, corpus, "dmv")
    assert (second.added, second.updated, second.unchanged, second.removed) == (0, 0, 3, 0)
    assert second.chunks_embedded == 0
    target = corpus / "dmv" / "insurance.txt"
    target.write_text(target.read_text(encoding="utf-8") + "\nPlates cost 25 dollars.\n", "utf-8")
    (corpus / "parking.html").unlink()
    (corpus / "new.txt").write_text("Fresh document\n\nBrand new content about boats.", "utf-8")
    third = sync_folder(store, embedder, cfg, corpus, "dmv")
    assert (third.added, third.updated, third.unchanged, third.removed) == (1, 1, 1, 1)
    paths = {d.path for d in store.list_documents("dmv")}
    assert paths == {"dmv/licenses.md", "dmv/insurance.txt", "new.txt"}
    assert store.sparse_search('"boats"', "dmv", 5)
    assert store.sparse_search('"dispute"', "dmv", 5) == []
    store.close()


def test_sync_skips_empty_files_and_reports_errors(cfg, tmp_path: Path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "empty.txt").write_text("   \n", encoding="utf-8")
    (folder / "broken.pdf").write_bytes(b"not a pdf")
    (folder / "ok.md").write_text("# Title\n\nSome body text that is long enough.", "utf-8")
    store = Store(cfg.resolve(cfg.paths.db_file))
    report = sync_folder(store, HashEmbedder(32), cfg, folder, "misc")
    assert report.added == 1
    assert report.skipped == 2
    assert len(report.errors) == 1 and "broken.pdf" in report.errors[0]
    store.close()
