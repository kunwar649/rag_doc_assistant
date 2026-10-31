"""LLM client with automatic fallback between free providers.

Groq and Gemini both expose OpenAI-compatible endpoints, so one SDK (openai)
talks to both; only the base URL, key and model name differ.

If the first provider fails (rate limit, timeout, outage, retired model), the
next one is tried. When streaming, fallback is only possible before the first
token arrives: after that the user has already seen part of an answer.
"""
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from openai import OpenAI, OpenAIError

from .config import Settings

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMUnavailableError(RuntimeError):
    """Every configured provider failed."""


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _usage(raw) -> dict | None:
    """Normalize an OpenAI-style usage object/dict to {"input": n, "output": n}."""
    if not raw:
        return None
    prompt, completion = _get(raw, "prompt_tokens"), _get(raw, "completion_tokens")
    if prompt is None and completion is None:
        return None
    return {"input": int(prompt or 0), "output": int(completion or 0)}


def _chunk_usage(chunk) -> dict | None:
    # OpenAI-style streams may put usage on the final chunk; Groq sends it under "x_groq".
    return _usage(getattr(chunk, "usage", None)) or _usage(_get(getattr(chunk, "x_groq", None) or {}, "usage"))


@dataclass
class Provider:
    name: str
    model: str
    client: object  # anything with .chat.completions.create(...) (OpenAI client or a test fake)
    extra: dict = field(default_factory=dict)  # provider-specific request options


class LLMClient:
    def __init__(self, providers: list[Provider], temperature: float = 0.1, max_tokens: int = 2048):
        if not providers:
            raise LLMUnavailableError(
                "No LLM provider configured. Set GROQ_API_KEY and/or GOOGLE_API_KEY in .env."
            )
        self.providers = providers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_provider: str | None = None
        self.last_usage: dict | None = None  # {"input": prompt tokens, "output": completion tokens}

    @property
    def default_model(self) -> str:
        return self.providers[0].model

    @classmethod
    def from_settings(cls, s: Settings) -> "LLMClient":
        providers = []
        if s.groq_api_key:
            extra = {}
            if s.groq_model.startswith("openai/gpt-oss"):
                # gpt-oss models reason before answering; "low" keeps answers fast
                # and leaves more of the free-tier token budget for the answer.
                extra["reasoning_effort"] = "low"
            providers.append(
                Provider("groq", s.groq_model, OpenAI(api_key=s.groq_api_key, base_url=GROQ_BASE_URL, timeout=s.llm_timeout, max_retries=1), extra)
            )
        if s.google_api_key:
            providers.append(
                Provider("gemini", s.gemini_model, OpenAI(api_key=s.google_api_key, base_url=GEMINI_BASE_URL, timeout=s.llm_timeout, max_retries=1))
            )
        return cls(providers, temperature=s.llm_temperature, max_tokens=s.llm_max_tokens)

    def _request(self, provider: Provider, messages: list[dict], stream: bool):
        return provider.client.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
            **provider.extra,
        )

    def complete(self, messages: list[dict]) -> str:
        errors = []
        self.last_usage = None
        for p in self.providers:
            try:
                resp = self._request(p, messages, stream=False)
                self.last_provider = f"{p.name}:{p.model}"
                self.last_usage = _usage(getattr(resp, "usage", None))
                return (resp.choices[0].message.content or "").strip()
            except OpenAIError as e:
                logger.warning("%s (%s) failed: %s", p.name, p.model, e)
                errors.append(f"{p.name}: {e}")
        raise LLMUnavailableError("All LLM providers failed: " + " | ".join(errors))

    def stream(self, messages: list[dict]) -> Iterator[str]:
        errors = []
        self.last_usage = None
        for p in self.providers:
            started = False
            try:
                for chunk in self._request(p, messages, stream=True):
                    usage = _chunk_usage(chunk)
                    if usage:
                        self.last_usage = usage
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content  # None for reasoning/role-only chunks
                    if text:
                        if not started:
                            started = True
                            self.last_provider = f"{p.name}:{p.model}"
                        yield text
                if started:
                    return
                errors.append(f"{p.name}: empty response")
            except OpenAIError as e:
                if started:
                    raise  # partial answer already sent; can't silently switch models
                logger.warning("%s (%s) failed: %s", p.name, p.model, e)
                errors.append(f"{p.name}: {e}")
        raise LLMUnavailableError("All LLM providers failed: " + " | ".join(errors))
