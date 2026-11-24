from convrag.chunking import chunk_text, index_text
from convrag.config import ChunkingConfig


def make_text(sentences: int, words: int = 8) -> str:
    return " ".join(
        " ".join(f"W{i}x{j}" for j in range(words - 1)) + f" end{i}." for i in range(sentences)
    )


def test_chunks_map_back_to_original_text():
    cfg = ChunkingConfig(max_words=20, min_words=5, overlap_sentences=1)
    text = make_text(12)
    chunks = chunk_text(text, cfg)
    assert len(chunks) > 1
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text.split()) <= cfg.max_words
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap_by_one_sentence():
    cfg = ChunkingConfig(max_words=16, min_words=4, overlap_sentences=1)
    text = make_text(6)
    chunks = chunk_text(text, cfg)
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert right.start < left.end
        shared = text[right.start : left.end]
        assert shared.endswith(".")


def test_no_overlap_when_disabled():
    cfg = ChunkingConfig(max_words=16, min_words=4, overlap_sentences=0)
    chunks = chunk_text(make_text(6), cfg)
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert right.start >= left.end


def test_long_sentence_is_split_on_word_boundaries():
    cfg = ChunkingConfig(max_words=10, min_words=2, overlap_sentences=0)
    text = " ".join(f"token{i}" for i in range(35))
    chunks = chunk_text(text, cfg)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 10, 5]
    assert "".join(c.text + " " for c in chunks).strip() == text


def test_small_tail_merges_into_previous_chunk():
    cfg = ChunkingConfig(max_words=20, min_words=10, overlap_sentences=0)
    text = make_text(4, words=8) + " Tiny tail with six words here."
    chunks = chunk_text(text, cfg)
    assert len(chunks) == 2
    assert chunks[-1].text.endswith("Tiny tail with six words here.")
    assert all(len(c.text.split()) >= cfg.min_words for c in chunks)


def test_empty_and_whitespace_text():
    cfg = ChunkingConfig()
    assert chunk_text("", cfg) == []
    assert chunk_text("   \n\n  ", cfg) == []


def test_index_text_prefixes_title():
    cfg = ChunkingConfig(title_prefix=True)
    assert index_text("Fees  and\nCosts", "pay   here\nnow", cfg) == "Fees and Costs\npay here now"
    cfg = ChunkingConfig(title_prefix=False)
    assert index_text("Fees", "pay here", cfg) == "pay here"
