from dataclasses import dataclass

from convrag.config import RewriteConfig
from convrag.llm import LLM, LLMError
from convrag.prompts import CONDENSE_SYSTEM, CONDENSE_USER, format_conversation
from convrag.text import normalize_ws

STRATEGIES = ("last_turn", "window", "history", "fusion", "rewrite")


@dataclass
class Formulation:
    strategy: str
    queries: list[str]
    rerank_query: str
    fallback: bool = False
    llm_calls: int = 0


def _window_text(history: list[tuple[str, str]], message: str, turns: int) -> str:
    recent = [normalize_ws(content) for _, content in history[-(turns - 1) :]] if turns > 1 else []
    return normalize_ws(" ".join([*recent, message]))


def _history_text(history: list[tuple[str, str]], message: str, max_chars: int) -> str:
    text = normalize_ws(" ".join([*(content for _, content in history), message]))
    if len(text) > max_chars:
        text = text[-max_chars:]
        cut = text.find(" ")
        if 0 <= cut < 40:
            text = text[cut + 1 :]
    return text


def _condense(history: list[tuple[str, str]], message: str, llm: LLM) -> str:
    conversation = format_conversation([*history, ("user", message)])
    prompt = CONDENSE_USER.format(conversation=conversation)
    completion = llm.complete(CONDENSE_SYSTEM, [{"role": "user", "content": prompt}])
    lines = [line.strip().strip('"') for line in completion.text.splitlines() if line.strip()]
    query = normalize_ws(lines[0]) if lines else ""
    if query.lower().startswith("standalone query:"):
        query = query.split(":", 1)[1].strip()
    return query[:400]


def formulate(
    history: list[tuple[str, str]],
    message: str,
    cfg: RewriteConfig,
    llm: LLM | None = None,
    strategy: str | None = None,
) -> Formulation:
    strategy = strategy or cfg.strategy
    message = normalize_ws(message)
    if strategy == "last_turn" or not history:
        return Formulation(strategy, [message], message)
    if strategy == "window":
        text = _window_text(history, message, cfg.window_turns)
        return Formulation(strategy, [text], text)
    if strategy == "history":
        text = _history_text(history, message, cfg.history_max_chars)
        return Formulation(strategy, [text], text)
    if strategy == "fusion":
        window = _window_text(history, message, cfg.window_turns)
        return Formulation(strategy, [window, message], window)
    if llm is None:
        window = _window_text(history, message, cfg.window_turns)
        return Formulation(strategy, [window], window, fallback=True)
    try:
        query = _condense(history, message, llm)
    except LLMError:
        query = ""
    if not query:
        window = _window_text(history, message, cfg.window_turns)
        return Formulation(strategy, [window], window, fallback=True, llm_calls=1)
    return Formulation(strategy, [query], query, llm_calls=1)
