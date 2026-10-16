"""Head-to-head benchmark: LLM-only vs System 1 + LLM, on the same questions.

- `run_pipeline`: the real BenefitPilot pipeline (any router).
- `run_llm_only`: one LLM call per question that both decides the route and
  writes the answer, given the same retrieved plan excerpts and, for a
  selected member, their year-to-date usage. No rules engine, no System 1.
- `judge_grounding`: an LLM-as-judge pass over generated answers, kept as a
  separate step so answers can be re-judged without regenerating them.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benefitpilot.evaluation.datasets import EvalItem
from benefitpilot.graph.pipeline import Pipeline
from benefitpilot.llm.client import LLMError, OllamaLLM
from benefitpilot.llm.prompts import RULES
from benefitpilot.retrieval.hybrid import HybridRetriever
from benefitpilot.rules.engine import fmt
from benefitpilot.rules.plan_schema import Plan
from benefitpilot.rules.summary import summarize
from benefitpilot.store import MemberStore

LLM_ONLY_ROUTES = [
    "coverage",
    "claim_check",
    "procedure",
    "refuse_medical",
    "emergency",
    "out_of_scope",
]


@dataclass
class BenchRecord:
    id: str
    split: str
    source: str
    pipeline: str
    expected_route: str
    route: str
    llm_called: bool
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    expected_citations: list[str] = field(default_factory=list)
    context: str = ""  # what the answer should be grounded in (for the judge)
    error: str | None = None
    judge_supported: bool | None = None
    judge_reason: str | None = None


def save(records: list[BenchRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(r)) for r in records) + "\n", encoding="utf-8")


def load(path: Path) -> list[BenchRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [BenchRecord(**json.loads(line)) for line in lines if line]


def _log(i: int, n: int, rec: BenchRecord) -> None:
    print(
        f"[{rec.pipeline}] {i}/{n} {rec.id}: {rec.route} (expected {rec.expected_route}) "
        f"{rec.latency_ms / 1000:.1f}s",
        flush=True,
    )


async def run_pipeline(pipeline: Pipeline, items: list[EvalItem], name: str) -> list[BenchRecord]:
    out = []
    for i, item in enumerate(items, 1):
        state, rec = await pipeline.ask(item.question, item.member_id)
        retrieval = state.get("retrieval")
        context = state.get("context_text") or ""
        if retrieval and not context:
            context = "\n\n".join(h.chunk.text for h in retrieval.hits)
        llm = rec.llm or {}
        r = BenchRecord(
            id=item.id,
            split=item.split,
            source=item.source,
            pipeline=name,
            expected_route=item.route,
            route=rec.route,
            llm_called=rec.llm is not None,
            llm_input_tokens=llm.get("input_tokens", 0),
            llm_output_tokens=llm.get("output_tokens", 0),
            latency_ms=rec.latency_ms["total"],
            cost_usd=rec.cost_usd,
            answer=rec.answer,
            citations=rec.citations,
            expected_citations=item.citations,
            context=context,
        )
        out.append(r)
        _log(i, len(items), r)
    return out


LLM_ONLY_SYSTEM = (
    "You are BenefitPilot, an assistant for members of a health and dental benefits plan.\n"
    "First decide how to handle the member's message, then reply. Routes:\n"
    "- coverage: what the plan covers in general\n"
    "- claim_check: whether a specific expense for this member will be paid, or their balances\n"
    "- procedure: how to use the plan (submitting claims, appeals, pre-authorization)\n"
    "- refuse_medical: the member asks for medical advice (treatment, medication, dose, what a "
    "symptom means, whether they need care); politely decline and suggest a health professional\n"
    "- emergency: a possibly life-threatening situation happening now; tell them to call 911\n"
    "- out_of_scope: anything else; say you can only help with their plan\n"
    + RULES
    + '\nReturn only JSON: {"route": "<one route>", "answer": "<your reply to the member>"}'
)


def _usage_lines(plan: Plan, store: MemberStore, member_id: str, today: dt.date) -> str:
    s = summarize(plan, store.snapshot(member_id, today.year), today)
    lines = [
        f"Member's plan: {plan.plan_id} ({plan.name}); "
        f"deductible left {fmt(s.deductible_remaining)}"
    ]
    for c in s.categories:
        limit = (
            f" of {fmt(c.annual_max)} annual maximum ({fmt(c.remaining)} left)"
            if c.annual_max
            else ""
        )
        lines.append(
            f"- {c.label} [{c.citation}]: plan pays {round(c.coinsurance * 100)}%; "
            f"plan has paid {fmt(c.used)} this year{limit}"
        )
    for g in s.groups:
        lines.append(f"- {g.label} combined: {fmt(g.used)} of {fmt(g.combined_max)} used")
    return "\n".join(lines)


async def run_llm_only(
    llm: OllamaLLM,
    retriever: HybridRetriever,
    items: list[EvalItem],
    plans: dict[str, Plan],
    store: MemberStore,
    today: dt.date,
    k: int = 3,
) -> list[BenchRecord]:
    out = []
    for i, item in enumerate(items, 1):
        start = time.perf_counter()
        plan = plans[store.plan_id(item.member_id)] if item.member_id else None
        result = await asyncio.to_thread(
            retriever.search, item.question, plan_id=plan.plan_id if plan else None, k=k
        )
        excerpts = "\n\n".join(
            f'<excerpt id="{h.citation}">\n{h.chunk.text}\n</excerpt>' for h in result.hits
        )
        usage = _usage_lines(plan, store, item.member_id, today) if plan else "No member selected."
        user = (
            f"Plan excerpts:\n{excerpts}\n\nMember's year-to-date usage:\n{usage}\n\n"
            f"<member_question>\n{item.question}\n</member_question>"
        )
        rec = BenchRecord(
            id=item.id,
            split=item.split,
            source=item.source,
            pipeline="llm_only",
            expected_route=item.route,
            route="invalid",
            llm_called=True,
            expected_citations=item.citations,
            context=excerpts + "\n\n" + usage,
        )
        try:
            r = await llm.complete(LLM_ONLY_SYSTEM, user, max_tokens=250, format="json")
            rec.llm_input_tokens, rec.llm_output_tokens = r.input_tokens, r.output_tokens
            try:
                parsed = json.loads(r.text)
                route = str(parsed.get("route", "")).strip()
                rec.route = route if route in LLM_ONLY_ROUTES else "invalid"
                rec.answer = str(parsed.get("answer", ""))
            except json.JSONDecodeError:
                rec.error, rec.answer = "invalid JSON", r.text
            rec.citations = [h.citation for h in result.hits if h.citation in rec.answer]
        except LLMError as e:
            rec.error = str(e)
        rec.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        out.append(rec)
        _log(i, len(items), rec)
    return out


JUDGE_SYSTEM = (
    "You check whether an assistant's answer about a health benefits plan is fully supported by "
    "the plan text it was given. Every statement about coverage, amounts, limits or procedures "
    "must be stated in, or follow directly from, the plan text. Ignore politeness and advice to "
    'contact someone. Return only JSON: {"supported": true or false, "reason": "<short reason>"}'
)


async def judge_grounding(judge: OllamaLLM, records: list[BenchRecord]) -> None:
    """Fill judge_supported / judge_reason on records whose answer came from an LLM."""
    # Skip answers already judged, so a later run (e.g. Jev) does not re-judge earlier ones.
    todo = [
        r for r in records if r.llm_called and r.answer and r.context and r.judge_supported is None
    ]
    for i, r in enumerate(todo, 1):
        user = f"Plan text:\n{r.context}\n\nAssistant's answer:\n{r.answer}"
        try:
            out = await judge.complete(JUDGE_SYSTEM, user, max_tokens=120, format="json")
            parsed = json.loads(out.text)
            r.judge_supported = bool(parsed.get("supported"))
            r.judge_reason = str(parsed.get("reason", ""))[:300]
        except (LLMError, json.JSONDecodeError) as e:
            r.judge_reason = f"judge failed: {e}"
        print(f"[judge] {i}/{len(todo)} {r.pipeline} {r.id}: {r.judge_supported}", flush=True)
