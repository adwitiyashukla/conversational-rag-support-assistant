import re
import string

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_LINE = re.compile(r"[^\n]+")
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(\s+)(?=[A-Z0-9\"'(\[])")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)

STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for from with without about as into
    is are was were be been being am do does did doing have has had having will would shall
    should can could may might must i me my we our you your he him his she her it its they
    them their this that these those there here what which who whom whose when where why how
    not no nor so than too very just also only own same such s t don can't isn't aren't
    """.split()
)


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for line in _LINE.finditer(text):
        start = line.start()
        segment = line.group()
        cursor = 0
        for match in _BOUNDARY.finditer(segment):
            end = match.start(1)
            spans.append((start + cursor, start + end))
            cursor = match.end(1)
        spans.append((start + cursor, start + len(segment)))
    cleaned = []
    for begin, end in spans:
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if end > begin:
            cleaned.append((begin, end))
    return cleaned


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    stripped = "".join(ch for ch in lowered if ch not in _PUNCT)
    without_articles = _ARTICLES.sub(" ", stripped)
    return " ".join(without_articles.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    counts: dict[str, int] = {}
    for token in ref:
        counts[token] = counts.get(token, 0) + 1
    common = 0
    for token in pred:
        if counts.get(token, 0) > 0:
            counts[token] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))
