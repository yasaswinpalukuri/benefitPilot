"""Provider-agnostic System 2 client: local Ollama (default) or Anthropic.

The rest of the code depends only on `LLMClient.complete(system, user)`.
Both adapters return token counts and a cost, so the audit log and the
Phase 7 benchmark see the same fields whichever model answered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import anthropic
import httpx

from benefitpilot.config import Settings


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete(self, system: str, user: str, *, max_tokens: int = 400) -> LLMResult: ...


class OllamaLLM:
    """Ollama /api/chat, non-streaming. Temperature 0 and a fixed seed for repeatable evals.

    Cost is reported as $0: it runs on hardware we already own (electricity ignored).
    """

    provider = "ollama"

    def __init__(self, base_url: str, model: str, *, timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 400, format: str | dict | None = None
    ) -> LLMResult:
        """`format`: Ollama structured output ("json" or a JSON schema); eval baselines only."""
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if format is not None:
            body["format"] = format
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=body)
                r.raise_for_status()
                data = r.json()
            text = data["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"Ollama chat failed at {self.base_url}: {e}") from e
        return LLMResult(
            text=text.strip(),
            model=data.get("model", self.model),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            cost_usd=0.0,
        )


# USD per million tokens (input, output), first-party API rates (cached 2026-10-06).
ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens
    "claude-opus-4-8": (5.00, 25.00),  # a possible server-side refusal fallback
}
# Models that accept server-side refusal fallbacks in the "default" form.
FALLBACK_CAPABLE = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}


class AnthropicLLM:
    """Claude via the official SDK (AsyncAnthropic).

    - No `temperature`: current models reject non-default sampling parameters.
    - Thinking can't be turned off on Opus 5.5; effort "low" keeps short
      phrasing tasks cheap and fast.
    - Server-side refusal fallbacks are on (`fallbacks="default"`), and a final
      `stop_reason == "refusal"` becomes an LLMError, so the graph falls back to
      its safe path.
    - The SDK retries 408/409/429/5xx and connection errors itself.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.model = model
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=3)

    async def complete(self, system: str, user: str, *, max_tokens: int = 400) -> LLMResult:
        kwargs = {}
        if self.model in FALLBACK_CAPABLE:
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        start = time.perf_counter()
        try:
            response = await self.client.beta.messages.create(
                model=self.model,
                # Thinking tokens count toward max_tokens; leave room beyond the answer.
                max_tokens=max(max_tokens, 16000),
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low"},
                **kwargs,
            )
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic request failed: {type(e).__name__}: {e}") from e
        if response.stop_reason == "refusal":
            raise LLMError("Anthropic model declined the request (stop_reason=refusal)")
        text = "".join(b.text for b in response.content if b.type == "text")
        served_by = response.model
        price_in, price_out = ANTHROPIC_PRICES.get(served_by, ANTHROPIC_PRICES["claude-opus-5-5"])
        usage = response.usage
        return LLMResult(
            text=text.strip(),
            model=served_by,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            cost_usd=(usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000,
        )


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise LLMError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
        return AnthropicLLM(
            settings.anthropic_model, api_key=settings.anthropic_api_key.get_secret_value()
        )
    return OllamaLLM(
        settings.ollama_base_url, settings.ollama_chat_model, timeout_s=settings.llm_timeout_s
    )
