"""JevRouter: System One via the official TypeSafe SDK (`typesafe-sdk`).

- The model is pinned (`jev-1.13.0`, not `jev-latest`): aliases move when a
  new release ships, which would silently change calibrated probabilities and
  invalidate tuned thresholds. Each response's versioned `model` is logged.
- Retries: the SDK's RetryPolicy (exponential backoff with jitter, honours
  `retry-after`, retries 408/429/5xx/529 and connection errors). Rate limits
  are per organization and not published, so we retry a little more than the
  default and cap the total time.
- Independent routing questions go out together in ONE request; the SDK
  evaluates them in parallel against the same state.
- Cost: Jev bills input tokens only ($0.042 per million for jev-1.13.0).
"""

from __future__ import annotations

import time

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score, TypeSafeError

from benefitpilot.routing.types import (
    ChoiceDecision,
    Decisions,
    NoulDecision,
    Question,
    RouterError,
    RouterState,
    ScoreDecision,
)

DEFAULT_MODEL = "jev-1.13.0"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
DEFAULT_RETRY = RetryPolicy(max_retries=4, backoff_initial=0.5, backoff_max=8.0, timeout=30.0)


def to_sdk(q: Question) -> Noul | Choice | Score:
    if q.kind == "noul":
        return Noul(instructions=q.instructions, criteria=q.criteria)
    if q.kind == "choice":
        return Choice(instructions=q.instructions, criteria=q.criteria)
    return Score(instructions=q.instructions, criteria=q.criteria)


class JevRouter:
    name = "jev"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: AsyncTypeSafeClient | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        timeout_s: float = 10.0,
    ):
        self.model = model
        self.client = client or AsyncTypeSafeClient(
            api_key=api_key, model=model, retry=retry, timeout=timeout_s
        )

    async def decide(self, state: RouterState, questions: list[Question]) -> Decisions:
        start = time.perf_counter()
        try:
            response = await self.client.system_one(
                state.as_state(),
                {q.id: to_sdk(q) for q in questions},
                model=self.model,
            )
        except TypeSafeError as e:
            raise RouterError(f"Jev request failed: {type(e).__name__}: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        answers = {}
        for q in questions:
            if q.kind == "noul":
                a = response.nouls[q.id]
                answers[q.id] = NoulDecision(p=a.noul)
            elif q.kind == "choice":
                a = response.choices[q.id]
                answers[q.id] = ChoiceDecision(
                    choice=a.choice, probabilities=dict(a.probabilities), confidence=a.confidence
                )
            else:
                a = response.scores[q.id]
                answers[q.id] = ScoreDecision(
                    score=a.score,
                    probabilities={int(k): v for k, v in a.probabilities.items()},
                    confidence=a.confidence,
                )
        tokens = response.usage.input_tokens
        return Decisions(
            answers=answers,
            router=self.name,
            model=response.model,
            latency_ms=round(latency_ms, 2),
            input_tokens=tokens,
            cost_usd=tokens * USD_PER_INPUT_TOKEN,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
