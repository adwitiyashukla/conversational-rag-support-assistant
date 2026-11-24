import re
from dataclasses import dataclass, field

from convrag.text import content_tokens, normalize_ws

STEM_PREFIX = 5
CITATION = re.compile(r"\s*\[(\d+)\]")
BOUNDARY = re.compile(r"[.!?]+[\"')]*(?:\s*\[\d+\])*(\s+)(?=[A-Z0-9\"'(])")


@dataclass
class Citation:
    sentence: int
    source: int
    overlap: float
    supported: bool


@dataclass
class VerifiedAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    cited_sources: list[int] = field(default_factory=list)
    grounded: bool = False
    dropped: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "grounded": self.grounded,
            "cited_sources": self.cited_sources,
            "dropped_citations": self.dropped,
            "citations": [
                {
                    "sentence": c.sentence,
                    "source": c.source,
                    "overlap": round(c.overlap, 3),
                    "supported": c.supported,
                }
                for c in self.citations
            ],
        }


def split_sentences(text: str) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    pieces = []
    cursor = 0
    for match in BOUNDARY.finditer(text):
        pieces.append(text[cursor : match.start(1)])
        cursor = match.end(1)
    pieces.append(text[cursor:])
    return [p.strip() for p in pieces if p.strip()]


def overlap_ratio(sentence: str, source: str) -> float:
    tokens = content_tokens(CITATION.sub("", sentence))
    if not tokens:
        return 0.0
    source_tokens = set(content_tokens(source))
    prefixes = {t[:STEM_PREFIX] for t in source_tokens if len(t) >= STEM_PREFIX}
    hits = sum(
        1
        for token in tokens
        if token in source_tokens or (len(token) >= STEM_PREFIX and token[:STEM_PREFIX] in prefixes)
    )
    return hits / len(tokens)


def verify(answer: str, sources: list[str], min_overlap: float) -> VerifiedAnswer:
    result = VerifiedAnswer(text="")
    kept_sentences = []
    for index, sentence in enumerate(split_sentences(answer)):
        refs = [int(m) for m in CITATION.findall(sentence)]
        cleaned = CITATION.sub("", sentence).strip()
        supported_refs = []
        for ref in refs:
            if not 1 <= ref <= len(sources):
                result.dropped += 1
                continue
            ratio = overlap_ratio(sentence, sources[ref - 1])
            ok = ratio >= min_overlap
            result.citations.append(Citation(index, ref, ratio, ok))
            if ok:
                if ref not in supported_refs:
                    supported_refs.append(ref)
            else:
                result.dropped += 1
        if supported_refs:
            cleaned = cleaned + " " + " ".join(f"[{r}]" for r in supported_refs)
        kept_sentences.append(cleaned)
    result.text = " ".join(kept_sentences).strip()
    supported = [c for c in result.citations if c.supported]
    cited_sentences = {c.sentence for c in result.citations}
    supported_sentences = {c.sentence for c in supported}
    result.cited_sources = sorted({c.source for c in supported})
    result.grounded = bool(supported) and cited_sentences == supported_sentences
    return result
