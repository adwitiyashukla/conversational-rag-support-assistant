from dataclasses import dataclass

from convrag.config import ChunkingConfig
from convrag.text import normalize_ws, split_sentences, word_count


@dataclass(frozen=True)
class Chunk:
    position: int
    start: int
    end: int
    text: str


def _split_long(text: str, span: tuple[int, int], max_words: int) -> list[tuple[int, int]]:
    begin, end = span
    if word_count(text[begin:end]) <= max_words:
        return [span]
    pieces: list[tuple[int, int]] = []
    cursor = begin
    words = 0
    piece_start = begin
    while cursor < end:
        while cursor < end and text[cursor].isspace():
            cursor += 1
        if cursor >= end:
            break
        word_end = cursor
        while word_end < end and not text[word_end].isspace():
            word_end += 1
        if words == 0:
            piece_start = cursor
        words += 1
        cursor = word_end
        if words == max_words:
            pieces.append((piece_start, cursor))
            words = 0
    if words:
        pieces.append((piece_start, cursor))
    return pieces


def chunk_text(text: str, cfg: ChunkingConfig) -> list[Chunk]:
    sentences: list[tuple[int, int]] = []
    for span in split_sentences(text):
        sentences.extend(_split_long(text, span, cfg.max_words))
    if not sentences:
        return []
    counts = [word_count(text[b:e]) for b, e in sentences]
    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(sentences):
        j = i
        words = 0
        while j < len(sentences) and (j == i or words + counts[j] <= cfg.max_words):
            words += counts[j]
            j += 1
        windows.append((i, j))
        if j >= len(sentences):
            break
        i += max(1, (j - i) - cfg.overlap_sentences)
    if len(windows) > 1:
        first, last = windows[-1]
        if sum(counts[first:last]) < cfg.min_words:
            prev_first, _ = windows[-2]
            windows = windows[:-2] + [(prev_first, last)]
    chunks = []
    for position, (first, last) in enumerate(windows):
        start = sentences[first][0]
        end = sentences[last - 1][1]
        chunks.append(Chunk(position=position, start=start, end=end, text=text[start:end]))
    return chunks


def index_text(title: str, chunk_text_value: str, cfg: ChunkingConfig) -> str:
    body = normalize_ws(chunk_text_value)
    if cfg.title_prefix and title:
        return f"{normalize_ws(title)}\n{body}"
    return body
