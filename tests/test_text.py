import pytest

from convrag.text import (
    content_tokens,
    exact_match,
    normalize_answer,
    split_sentences,
    token_f1,
    tokenize,
)


def test_split_sentences_keeps_exact_offsets():
    text = "First one. Second one!  Third?\n\nA new line here. And more"
    spans = split_sentences(text)
    pieces = [text[a:b] for a, b in spans]
    assert pieces == ["First one.", "Second one!", "Third?", "A new line here.", "And more"]


def test_split_sentences_does_not_break_on_lowercase_continuation():
    text = "The fee is 64 dollars. it covers eight years."
    assert len(split_sentences(text)) == 1


def test_tokenize_and_content_tokens():
    assert tokenize("Don't Pay the $64 Fee!") == ["don't", "pay", "the", "64", "fee"]
    assert content_tokens("the fee is 64 dollars") == ["fee", "64", "dollars"]


def test_normalize_answer_strips_articles_and_punctuation():
    assert normalize_answer("The DMV, will suspend it.") == "dmv will suspend it"


def test_token_f1_and_exact_match():
    assert token_f1("renew online", "you can renew online") == pytest.approx(2 / 3)
    assert token_f1("", "") == 1.0
    assert token_f1("nothing shared", "renew online") == 0.0
    assert exact_match("The license.", "license") == 1.0
    assert exact_match("a license", "two licenses") == 0.0
