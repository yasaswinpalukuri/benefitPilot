"""The DecisionRouter interface shared by JevRouter and FallbackRouter.

Questions are defined once, in TypeSafe's primitive vocabulary (Noul, Choice,
Score), so both routers answer exactly the same typed questions and their
outputs are directly comparable in the Phase 7 benchmark.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    """One typed judgment. `id` is for code; the meaning lives in instructions/criteria."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["noul", "choice", "score"]
    instructions: str | dict[str, Any]
    # noul: {"true": ..., "false": ...} | choice: {option: description} | score: [levels]
    criteria: dict[str, Any] | list[Any] | None = None

    @property
    def options(self) -> list[str]:
        if self.kind != "choice" or not isinstance(self.criteria, dict):
            raise TypeError(f"{self.id} is not a choice question")
        return list(self.criteria)


class RouterState(BaseModel):
    """What the router sees. Only non-empty fields are sent."""

    member_message: str
    plan: str | None = None  # e.g. "PLAN-A (Bronze)"; never names or other PII
    assistant_answer: str | None = None  # output check only
    retrieved_plan_text: str | None = None  # output check only

    def as_state(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v}


class NoulDecision(BaseModel):
    kind: Literal["noul"] = "noul"
    p: float = Field(ge=0, le=1)  # probability of yes


class ChoiceDecision(BaseModel):
    kind: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def top_p(self) -> float:
        return self.probabilities[self.choice]


class ScoreDecision(BaseModel):
    kind: Literal["score"] = "score"
    score: float
    probabilities: dict[int, float]  # level index -> probability
    confidence: float


Decision = Annotated[NoulDecision | ChoiceDecision | ScoreDecision, Field(discriminator="kind")]


class Decisions(BaseModel):
    """Raw answers plus provenance. Stored verbatim in the audit log so thresholds
    can be re-tuned later without re-running inference."""

    answers: dict[str, Decision]
    router: str  # "jev", "fallback", "fallback (jev failed)"
    model: str | None = None  # versioned model ID that answered, e.g. "jev-1.13.0"
    latency_ms: float
    input_tokens: int | None = None
    cost_usd: float = 0.0
    error: str | None = None  # set when a failover happened

    def noul(self, qid: str) -> NoulDecision:
        return self._get(qid, NoulDecision)

    def choice(self, qid: str) -> ChoiceDecision:
        return self._get(qid, ChoiceDecision)

    def score(self, qid: str) -> ScoreDecision:
        return self._get(qid, ScoreDecision)

    def _get(self, qid, cls):
        answer = self.answers[qid]
        if not isinstance(answer, cls):
            raise TypeError(f"{qid} is a {answer.kind}, not {cls.__name__}")
        return answer


class RouterError(RuntimeError):
    pass


class DecisionRouter(Protocol):
    name: str

    async def decide(self, state: RouterState, questions: list[Question]) -> Decisions: ...
