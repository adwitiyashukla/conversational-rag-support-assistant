from convrag.config import RewriteConfig
from convrag.llm import FakeLLM, LLMError
from convrag.rewrite import formulate

HISTORY = [
    ("user", "My insurance ended so what should I do"),
    ("agent", "You will need to get insurance or we will suspend your registration"),
    ("user", "How long do I have"),
    ("agent", "You have ten days from the lapse date"),
]
MESSAGE = "And what does the letter say?"


def test_last_turn_uses_only_the_message():
    cfg = RewriteConfig(strategy="last_turn")
    result = formulate(HISTORY, MESSAGE, cfg)
    assert result.queries == [MESSAGE]
    assert result.rerank_query == MESSAGE


def test_window_includes_recent_turns_only():
    cfg = RewriteConfig(strategy="window", window_turns=3)
    result = formulate(HISTORY, MESSAGE, cfg)
    assert result.queries == [
        "How long do I have You have ten days from the lapse date And what does the letter say?"
    ]
    single = formulate(HISTORY, MESSAGE, RewriteConfig(strategy="window", window_turns=1))
    assert single.queries == [MESSAGE]


def test_history_is_truncated_from_the_front():
    cfg = RewriteConfig(strategy="history", history_max_chars=60)
    result = formulate(HISTORY, MESSAGE, cfg)
    query = result.queries[0]
    assert query.endswith(MESSAGE)
    assert len(query) <= 60
    assert not query.startswith(" ")
    full = formulate(HISTORY, MESSAGE, RewriteConfig(strategy="history", history_max_chars=10000))
    assert full.queries[0].startswith("My insurance ended")


def test_fusion_sends_window_and_message():
    cfg = RewriteConfig(strategy="fusion", window_turns=2)
    result = formulate(HISTORY, MESSAGE, cfg)
    assert result.queries == ["You have ten days from the lapse date " + MESSAGE, MESSAGE]
    assert result.rerank_query == result.queries[0]


def test_rewrite_calls_the_llm_and_cleans_the_output():
    llm = FakeLLM(responder=lambda system, messages: '"What does the lapse letter say?"\n')
    result = formulate(HISTORY, MESSAGE, RewriteConfig(strategy="rewrite"), llm)
    assert result.queries == ["What does the lapse letter say?"]
    assert result.llm_calls == 1 and not result.fallback
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "ten days" in prompt and MESSAGE in prompt


def test_rewrite_falls_back_to_window_on_empty_or_failing_llm():
    cfg = RewriteConfig(strategy="rewrite", window_turns=2)
    empty = formulate(HISTORY, MESSAGE, cfg, FakeLLM(responder=lambda s, m: "   "))
    assert empty.fallback and empty.queries == ["You have ten days from the lapse date " + MESSAGE]

    def boom(system, messages):
        raise LLMError("down")

    failing = formulate(HISTORY, MESSAGE, cfg, FakeLLM(responder=boom))
    assert failing.fallback and failing.strategy == "rewrite"
    assert formulate(HISTORY, MESSAGE, cfg, None).fallback


def test_first_turn_never_needs_rewriting():
    llm = FakeLLM()
    result = formulate([], "How do I renew?", RewriteConfig(strategy="rewrite"), llm)
    assert result.queries == ["How do I renew?"]
    assert llm.calls == []
