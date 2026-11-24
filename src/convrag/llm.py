import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from convrag.config import LLMConfig


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict]) -> Completion: ...

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...


def _sse_lines(response: httpx.Response) -> Iterator[dict]:
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


class OpenAICompatibleLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.name = cfg.model
        self.client = httpx.Client(timeout=httpx.Timeout(cfg.timeout_seconds, connect=10.0))

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _payload(self, system: str, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "stream": stream,
        }

    def _url(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/chat/completions"

    def complete(self, system: str, messages: list[dict]) -> Completion:
        try:
            response = self.client.post(
                self._url(), headers=self._headers(), json=self._payload(system, messages, False)
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"llm request failed: {exc}") from exc
        body = response.json()
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected llm response: {body}") from exc
        usage = body.get("usage") or {}
        return Completion(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=str(body.get("model") or self.cfg.model),
        )

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        try:
            with self.client.stream(
                "POST",
                self._url(),
                headers=self._headers(),
                json=self._payload(system, messages, True),
            ) as response:
                response.raise_for_status()
                for event in _sse_lines(response):
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise LLMError(f"llm stream failed: {exc}") from exc


class AnthropicLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.name = cfg.model
        self.client = httpx.Client(timeout=httpx.Timeout(cfg.timeout_seconds, connect=10.0))

    def _headers(self) -> dict:
        if not self.cfg.api_key:
            raise LLMError(f"{self.cfg.api_key_env} is not set")
        return {
            "Content-Type": "application/json",
            "x-api-key": self.cfg.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _payload(self, system: str, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.cfg.model,
            "system": system,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "stream": stream,
        }

    def _url(self) -> str:
        base = self.cfg.base_url.rstrip("/")
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def complete(self, system: str, messages: list[dict]) -> Completion:
        try:
            response = self.client.post(
                self._url(), headers=self._headers(), json=self._payload(system, messages, False)
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"llm request failed: {exc}") from exc
        body = response.json()
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        return Completion(
            text=text.strip(),
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            model=str(body.get("model") or self.cfg.model),
        )

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        try:
            with self.client.stream(
                "POST",
                self._url(),
                headers=self._headers(),
                json=self._payload(system, messages, True),
            ) as response:
                response.raise_for_status()
                for event in _sse_lines(response):
                    if event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]
        except httpx.HTTPError as exc:
            raise LLMError(f"llm stream failed: {exc}") from exc


def _default_responder(system: str, messages: list[dict]) -> str:
    user = messages[-1]["content"] if messages else ""
    if "standalone" in system.lower():
        turns = [line for line in user.splitlines() if line.startswith("user:")]
        return turns[-1].split(":", 1)[1].strip() if turns else user
    marker = "[1]"
    start = user.find(marker)
    if start < 0:
        return "NOT_IN_SOURCES"
    snippet = user[start + len(marker) :].split("\n", 1)[0].strip()
    first_sentence = snippet.split(". ")[0].strip().rstrip(".")
    return f"{first_sentence}. [1]"


@dataclass
class FakeLLM:
    responder: Callable[[str, list[dict]], str] = _default_responder
    name: str = "fake"
    calls: list[dict] = field(default_factory=list)

    def complete(self, system: str, messages: list[dict]) -> Completion:
        self.calls.append({"system": system, "messages": messages})
        text = self.responder(system, messages)
        return Completion(text=text, prompt_tokens=len(system.split()), model=self.name)

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        text = self.complete(system, messages).text
        words = text.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else word + " "


def build_llm(cfg: LLMConfig) -> LLM:
    if cfg.provider == "fake":
        return FakeLLM()
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg)
    return OpenAICompatibleLLM(cfg)
