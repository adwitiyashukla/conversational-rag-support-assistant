import re

ABSTAIN_TOKEN = "NOT_IN_SOURCES"
ABSTAIN_PATTERN = re.compile(r"not[\s_]*in[\s_]*sources", re.IGNORECASE)

CONDENSE_SYSTEM = (
    "You turn the latest user message of a support conversation into one standalone search "
    "query. Resolve pronouns and references using the earlier turns, keep every concrete "
    "detail the user gave, and do not answer the question. Reply with the query only."
)

CONDENSE_USER = "Conversation:\n{conversation}\n\nStandalone query:"

ANSWER_SYSTEM = (
    "You are a support assistant. Answer the user's latest message using only the numbered "
    "sources. Write one to three plain sentences. End every sentence that uses a source with "
    "its citation, like [2]. Do not mention sources that you did not use. If the sources do "
    f"not contain the answer, reply with exactly {ABSTAIN_TOKEN} and nothing else."
)

ANSWER_USER = (
    "Sources:\n{sources}\n\nConversation so far:\n{conversation}\n\n"
    "Latest user message: {message}\n\nAnswer:"
)


def format_conversation(turns: list[tuple[str, str]]) -> str:
    if not turns:
        return "(no earlier turns)"
    return "\n".join(f"{role}: {content}" for role, content in turns)


def format_sources(sources: list[tuple[str, str]]) -> str:
    return "\n".join(f"[{i}] ({title}) {text}" for i, (title, text) in enumerate(sources, start=1))
