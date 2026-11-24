from convrag.citations import overlap_ratio, split_sentences, verify

SOURCES = [
    "If your insurance ends, the DMV will suspend your registration and your license.",
    "You can dispute a parking ticket online with the ticket number and your plate.",
]


def test_split_sentences_keeps_trailing_citations_with_their_sentence():
    text = "The DMV suspends your registration. [1] You can dispute online. [2] Thanks!"
    assert split_sentences(text) == [
        "The DMV suspends your registration. [1]",
        "You can dispute online. [2]",
        "Thanks!",
    ]
    assert split_sentences("Pay within 30 days [2]. Then relax.") == [
        "Pay within 30 days [2].",
        "Then relax.",
    ]


def test_overlap_ratio_ignores_citation_markers_and_stopwords():
    assert overlap_ratio("The DMV will suspend your registration [1].", SOURCES[0]) == 1.0
    assert overlap_ratio("Boats are fun [1]", SOURCES[0]) == 0.0
    assert overlap_ratio("the and of", SOURCES[0]) == 0.0


def test_overlap_ratio_matches_inflected_forms_by_prefix():
    assert overlap_ratio("Registration suspensions and licenses", SOURCES[0]) == 1.0
    assert overlap_ratio("Disputing tickets", SOURCES[1]) == 1.0
    assert overlap_ratio("ends plate", SOURCES[0]) == 0.5


def test_verify_keeps_supported_and_drops_unsupported_citations():
    answer = (
        "If your insurance ends the DMV will suspend your registration and license. [1] "
        "You should buy a boat instead. [2] Disputes happen online with the ticket number [2]."
    )
    result = verify(answer, SOURCES, min_overlap=0.5)
    assert result.text == (
        "If your insurance ends the DMV will suspend your registration and license. [1] "
        "You should buy a boat instead. Disputes happen online with the ticket number. [2]"
    )
    assert result.cited_sources == [1, 2]
    assert result.dropped == 1
    assert not result.grounded
    supported = [c for c in result.citations if c.supported]
    assert [(c.sentence, c.source) for c in supported] == [(0, 1), (2, 2)]


def test_verify_marks_fully_supported_answers_grounded():
    answer = "The DMV will suspend your registration and license when insurance ends. [1]"
    result = verify(answer, SOURCES, min_overlap=0.5)
    assert result.grounded
    assert result.cited_sources == [1]
    assert result.text == answer


def test_verify_handles_invalid_indices_and_missing_citations():
    result = verify("Something made up. [7] Another claim with no source.", SOURCES, 0.3)
    assert result.text == "Something made up. Another claim with no source."
    assert result.dropped == 1
    assert not result.grounded
    assert result.cited_sources == []
    assert verify("", SOURCES, 0.3).text == ""
