import pytest

from convrag.llm import FakeLLM, LLMError
from convrag.prompts import ABSTAIN_TOKEN


def test_respond_answers_with_verified_citation_and_persists(runtime):
    result = runtime.chat.respond("my insurance ended, what happens now?", "dmv")
    assert result.session_id
    assert not result.abstained
    assert result.grounded
    assert result.answer.endswith("[1]")
    assert result.hits[0].chunk.path == "dmv/insurance.txt"
    assert result.verified.cited_sources == [1]
    messages = runtime.store.get_messages(result.session_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].payload["grounded"] is True
    assert messages[1].payload["sources"][0]["path"] == "dmv/insurance.txt"
    names = [s.name for s in result.trace.spans]
    assert names == ["rewrite", "sparse", "dense", "fuse", "rerank", "generate", "verify"]
    traces = runtime.store.recent_traces(5)
    assert traces[0]["kind"] == "chat" and traces[0]["session_id"] == result.session_id


def test_follow_up_uses_session_history_in_the_query(runtime):
    first = runtime.chat.respond("My insurance ended so what should I do", "dmv")
    second = runtime.chat.respond("How long do I have?", "dmv", first.session_id)
    assert second.formulation.strategy == "window"
    assert "insurance" in second.formulation.queries[0]
    assert second.hits[0].chunk.path == "dmv/insurance.txt"
    assert len(runtime.store.get_messages(first.session_id)) == 4
    prompt = runtime.llm.calls[-1]["messages"][0]["content"]
    assert "My insurance ended" in prompt


def test_abstains_when_nothing_is_retrieved(runtime):
    result = runtime.chat.respond("anything", "empty-collection")
    assert result.abstained and result.abstain_reason == "no_results"
    assert result.answer == runtime.cfg.generation.abstain_message
    assert runtime.llm.calls == []


def test_abstains_when_rerank_score_is_too_low(runtime):
    runtime.cfg.reranker.min_score = 0.99
    result = runtime.chat.respond("tell me about boats and oceans", "dmv")
    assert result.abstain_reason == "low_relevance"
    assert runtime.llm.calls == []


@pytest.mark.parametrize(
    "reply", [ABSTAIN_TOKEN, "Not_in_Sources.", "not in sources", " NOT IN_SOURCES "]
)
def test_abstains_when_model_says_so(runtime, reply):
    runtime.llm.responder = lambda system, messages: reply
    result = runtime.chat.respond("renew my license", "dmv")
    assert result.abstain_reason == "model_abstained"
    assert result.answer == runtime.cfg.generation.abstain_message
    assert not result.grounded


def test_unverifiable_answer_is_kept_but_not_grounded(runtime):
    runtime.llm.responder = lambda system, messages: "Buy a boat and sail away. [1]"
    result = runtime.chat.respond("renew my license", "dmv")
    assert not result.abstained
    assert not result.grounded
    assert result.answer == "Buy a boat and sail away."
    assert result.verified.dropped == 1


def test_stream_emits_events_in_order(runtime):
    events = list(runtime.chat.respond_stream("dispute a parking ticket", "dmv"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[1] == "queries" and kinds[2] == "sources"
    assert "token" in kinds and kinds[-1] == "done"
    done = events[-1]
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed.strip().startswith(done["answer"].split(" [")[0][:20])
    assert done["sources"][0]["path"] == "parking.html"
    assert done["session_id"] == events[0]["session_id"]
    assert len(runtime.store.get_messages(done["session_id"])) == 2


def test_llm_errors_propagate(runtime):
    def boom(system, messages):
        raise LLMError("offline")

    runtime.llm.responder = boom
    with pytest.raises(LLMError):
        runtime.chat.respond("renew my license", "dmv")
    assert runtime.store.recent_traces(5) == []


def test_metrics_render_after_requests(runtime):
    runtime.chat.respond("renew my license", "dmv")
    runtime.chat.respond("nothing", "missing")
    text = runtime.metrics.render()
    assert 'convrag_requests_total{kind="chat"} 2' in text
    assert 'convrag_abstentions_total{reason="no_results"} 1' in text
    assert 'convrag_stage_ms_count{kind="chat",stage="generate"} 1' in text
    assert "# TYPE convrag_request_ms histogram" in text


def test_fake_llm_default_responder_reads_first_source():
    llm = FakeLLM()
    prompt = [{"role": "user", "content": "Sources:\n[1] (T) Alpha beta. Gamma.\n"}]
    assert llm.complete("answer", prompt).text == "(T) Alpha beta. [1]"
    empty = [{"role": "user", "content": "no sources"}]
    assert llm.complete("answer", empty).text == ABSTAIN_TOKEN
