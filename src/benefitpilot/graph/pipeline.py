"""The BenefitPilot request pipeline as a LangGraph state machine.

    context -> route --+--> canned (emergency / refuse / out_of_scope / escalate) -> END
                       +--> claim  ---+                     (no LLM on these paths)
                       +--> retrieve -+--> generate -> guard -> output_check -> END
                                      +--> END  (missing member / details, or not in the plan)

- System 1 (`route`) makes one request: the routing questions, plus the
  speculative claim-argument questions when a member is selected.
- The rules engine computes all money; System 2 only phrases it. `guard`
  rejects any dollar amount or percentage the engine didn't produce and falls
  back to a deterministic template.
- `output_check` is the second System 1 request. It can turn an answer into
  a refusal, a template, or an escalation, never the reverse.
- `Pipeline.ask` wraps the graph and writes one audit record per request,
  whichever path ran.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from benefitpilot.audit.log import AuditRecord, AuditStore, new_audit_id
from benefitpilot.guardrails import messages
from benefitpilot.guardrails.grounding import (
    cited_ids,
    claim_inconsistencies,
    unsupported_numbers,
)
from benefitpilot.llm.client import LLMClient, LLMError, LLMResult
from benefitpilot.llm.prompts import (
    CLAIM_SYSTEM,
    COVERAGE_SYSTEM,
    claim_facts,
    claim_prompt,
    coverage_prompt,
)
from benefitpilot.retrieval.hybrid import HybridRetriever, RetrievalResult
from benefitpilot.routing.claim_args import ClaimArgs, claim_questions, read_claim_args
from benefitpilot.routing.policy import (
    OutputVerdict,
    RouteDecision,
    Thresholds,
    check_output,
    route,
)
from benefitpilot.routing.questions import (
    ANSWER_CONTAINS_MEDICAL_ADVICE,
    OUTPUT_CHECK_QUESTIONS,
    ROUTING_QUESTIONS,
)
from benefitpilot.routing.types import DecisionRouter, Decisions, RouterState
from benefitpilot.rules.engine import check_claim
from benefitpilot.rules.models import ClaimInput, ClaimResult
from benefitpilot.rules.plan_schema import Plan
from benefitpilot.store import MemberStore

CANNED_ROUTES = {"emergency", "refuse_medical", "out_of_scope", "escalate"}
RETRIEVAL_K = 3  # small context: the default model is a 7B model on a CPU


class PipelineState(TypedDict, total=False):
    audit_id: str
    question: str
    member_id: str | None
    plan_id: str | None
    plan_label: str | None
    decisions: Decisions
    route_decision: RouteDecision
    final_route: str
    claim_args: ClaimArgs
    rule_result: ClaimResult
    retrieval: RetrievalResult
    prompt: tuple[str, str]  # (system, user)
    context_text: str  # what the answer must be grounded in
    allowed_citations: list[str]
    template: str  # deterministic fallback for claim answers
    llm_result: LLMResult
    draft: str
    output_decisions: Decisions
    output_verdict: OutputVerdict
    answer: str
    citations: list[str]
    escalation_reason: str | None
    guardrails: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], operator.or_]


@dataclass
class Deps:
    router: DecisionRouter
    retriever: HybridRetriever
    llm: LLMClient
    store: MemberStore
    plans: dict[str, Plan]
    thresholds: Thresholds
    audit: AuditStore
    today: Callable[[], dt.date] = field(default=dt.date.today)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class Pipeline:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.graph = self._build()

    # --- nodes ----------------------------------------------------------------

    async def context(self, s: PipelineState) -> dict:
        if not s.get("member_id"):
            return {"plan_id": None, "plan_label": None}
        plan = self.deps.plans[self.deps.store.plan_id(s["member_id"])]
        return {"plan_id": plan.plan_id, "plan_label": f"{plan.plan_id} ({plan.name})"}

    async def route_node(self, s: PipelineState) -> dict:
        start = time.perf_counter()
        questions = list(ROUTING_QUESTIONS)
        if s.get("plan_id"):
            questions += claim_questions(self.deps.plans[s["plan_id"]], s["question"])
        decisions = await self.deps.router.decide(
            RouterState(member_message=s["question"], plan=s.get("plan_label")), questions
        )
        rd = route(decisions, self.deps.thresholds)
        return {
            "decisions": decisions,
            "route_decision": rd,
            "final_route": rd.route,
            "timings": {"route": _ms(start)},
        }

    async def canned(self, s: PipelineState) -> dict:
        r = s["route_decision"]
        text = {
            "emergency": messages.EMERGENCY,
            "refuse_medical": messages.REFUSE_MEDICAL,
            "out_of_scope": messages.OUT_OF_SCOPE,
        }.get(r.route)
        if r.route == "escalate":
            return {
                "answer": messages.escalated(s["audit_id"]),
                "citations": [],
                "escalation_reason": r.reason,
            }
        return {"answer": text, "citations": []}

    async def claim(self, s: PipelineState) -> dict:
        start = time.perf_counter()
        if not s.get("plan_id"):
            return {"answer": messages.NEED_MEMBER, "citations": []}
        args = read_claim_args(s["decisions"], s["question"])
        if not args.complete:
            return {"answer": messages.NEED_CLAIM_DETAILS, "citations": [], "claim_args": args}
        plan = self.deps.plans[s["plan_id"]]
        today = self.deps.today()
        snapshot = self.deps.store.snapshot(s["member_id"], today.year)
        result = check_claim(
            plan,
            snapshot,
            ClaimInput(
                member_id=s["member_id"], category=args.category, amount=args.amount, date=today
            ),
        )
        label = (
            plan.categories[args.category].label
            if args.category in plan.categories
            else args.category.replace("_", " ")
        )
        return {
            "claim_args": args,
            "rule_result": result,
            "prompt": (CLAIM_SYSTEM, claim_prompt(s["question"], result)),
            "context_text": claim_facts(result),
            "allowed_citations": result.citations,
            "template": messages.claim_template(result, label),
            "timings": {"rules": _ms(start)},
        }

    async def retrieve(self, s: PipelineState) -> dict:
        start = time.perf_counter()
        result = await asyncio.to_thread(
            self.deps.retriever.search, s["question"], plan_id=s.get("plan_id"), k=RETRIEVAL_K
        )
        out: dict[str, Any] = {"retrieval": result, "timings": {"retrieve": _ms(start)}}
        if result.degraded:
            out["guardrails"] = [f"retrieval_degraded: {result.degraded}"]
        if not result.relevant:
            out |= {"answer": messages.NOT_FOUND, "citations": []}
            return out
        return out | {
            "prompt": (
                COVERAGE_SYSTEM,
                coverage_prompt(s["question"], result.hits, s.get("plan_label")),
            ),
            "context_text": "\n\n".join(h.chunk.text for h in result.hits),
            "allowed_citations": result.citations,
        }

    async def generate(self, s: PipelineState) -> dict:
        start = time.perf_counter()
        system, user = s["prompt"]
        try:
            result = await self.deps.llm.complete(system, user)
        except LLMError as e:
            note = [f"llm_failed: {e}"]
            if s.get("template"):  # claim: the engine result can be explained without an LLM
                return {
                    "draft": s["template"],
                    "guardrails": note + ["used deterministic claim template"],
                    "timings": {"generate": _ms(start)},
                }
            return {
                "answer": messages.escalated(s["audit_id"]),
                "citations": [],
                "final_route": "escalate",
                "escalation_reason": f"LLM unavailable: {e}",
                "guardrails": note,
                "timings": {"generate": _ms(start)},
            }
        return {"llm_result": result, "draft": result.text, "timings": {"generate": _ms(start)}}

    async def guard(self, s: PipelineState) -> dict:
        draft, notes = s["draft"], []
        if s.get("template"):
            r = s["rule_result"]
            problems = [
                f"{n} not produced by the rules engine"
                for n in unsupported_numbers(draft, s["context_text"])
            ]
            problems += claim_inconsistencies(draft, r.outcome, r.covered_amount, r.member_pays)
            if problems:
                notes.append(f"money_guard: {'; '.join(problems)}; used template")
                draft = s["template"]
        cited = cited_ids(draft, s["allowed_citations"])
        if not cited:
            notes.append("no valid inline citation; listing the sections the answer was based on")
            cited = list(s["allowed_citations"])
        return {"draft": draft, "citations": cited, "guardrails": notes}

    async def output_check(self, s: PipelineState) -> dict:
        start = time.perf_counter()
        decisions = await self.deps.router.decide(
            RouterState(
                member_message=s["question"],
                assistant_answer=s["draft"],
                retrieved_plan_text=s["context_text"],
            ),
            OUTPUT_CHECK_QUESTIONS,
        )
        verdict = check_output(decisions, self.deps.thresholds)
        out: dict[str, Any] = {
            "output_decisions": decisions,
            "output_verdict": verdict,
            "timings": {"output_check": _ms(start)},
        }
        if verdict.passed:
            return out | {"answer": s["draft"]}
        notes = [f"output_check failed: {', '.join(verdict.reasons)}"]
        if any(r.startswith(ANSWER_CONTAINS_MEDICAL_ADVICE.id) for r in verdict.reasons):
            return out | {
                "answer": messages.REFUSE_MEDICAL,
                "citations": [],
                "final_route": "refuse_medical",
                "guardrails": notes,
            }
        if s.get("template"):
            return out | {
                "answer": s["template"],
                "citations": s["allowed_citations"],
                "guardrails": notes + ["used deterministic claim template"],
            }
        return out | {
            "answer": messages.escalated(s["audit_id"]),
            "citations": [],
            "final_route": "escalate",
            "escalation_reason": "answer not supported by the plan text",
            "guardrails": notes,
        }

    # --- edges ----------------------------------------------------------------

    @staticmethod
    def _after_route(s: PipelineState) -> str:
        r = s["route_decision"].route
        if r in CANNED_ROUTES:
            return "canned"
        return "claim" if r == "claim_check" else "retrieve"

    @staticmethod
    def _answered_or(next_node: str) -> Callable[[PipelineState], str]:
        return lambda s: END if s.get("answer") else next_node

    def _build(self):
        g = StateGraph(PipelineState)
        g.add_node("context", self.context)
        g.add_node("route", self.route_node)
        g.add_node("canned", self.canned)
        g.add_node("claim", self.claim)
        g.add_node("retrieve", self.retrieve)
        g.add_node("generate", self.generate)
        g.add_node("guard", self.guard)
        g.add_node("output_check", self.output_check)
        g.add_edge(START, "context")
        g.add_edge("context", "route")
        g.add_conditional_edges("route", self._after_route, ["canned", "claim", "retrieve"])
        g.add_edge("canned", END)
        g.add_conditional_edges("claim", self._answered_or("generate"), ["generate", END])
        g.add_conditional_edges("retrieve", self._answered_or("generate"), ["generate", END])
        g.add_conditional_edges("generate", self._answered_or("guard"), ["guard", END])
        g.add_edge("guard", "output_check")
        g.add_edge("output_check", END)
        return g.compile()

    # --- entry point ----------------------------------------------------------

    async def ask(
        self, question: str, member_id: str | None = None
    ) -> tuple[PipelineState, AuditRecord]:
        start = time.perf_counter()
        initial: PipelineState = {
            "audit_id": new_audit_id(),
            "question": question,
            "member_id": member_id,
            "guardrails": [],
            "timings": {},
        }
        state: PipelineState = await self.graph.ainvoke(initial)
        timings = dict(state.get("timings", {})) | {"total": _ms(start)}
        state["timings"] = timings
        record = self._audit_record(state)
        self.deps.audit.write(record)
        return state, record

    def _audit_record(self, s: PipelineState) -> AuditRecord:
        d = s["decisions"]
        out = s.get("output_decisions")
        llm = s.get("llm_result")
        cost = d.cost_usd + (out.cost_usd if out else 0.0) + (llm.cost_usd if llm else 0.0)
        retrieval = s.get("retrieval")
        args = s.get("claim_args")
        result = s.get("rule_result")
        verdict = s.get("output_verdict")
        return AuditRecord(
            audit_id=s["audit_id"],
            member_id=s.get("member_id"),
            question=s["question"],
            route=s["final_route"],
            route_reason=s["route_decision"].reason,
            router={
                "name": d.router,
                "model": d.model,
                "error": d.error,
                "latency_ms": d.latency_ms,
                "input_tokens": d.input_tokens,
                "cost_usd": d.cost_usd,
                "decisions": d.model_dump(mode="json")["answers"],
                "initial_route": s["route_decision"].route,
            },
            claim_args=(
                {
                    "category": args.category,
                    "amount": str(args.amount) if args.amount is not None else None,
                    "category_p": args.category_p,
                    "amount_p": args.amount_p,
                }
                if args
                else None
            ),
            retrieval=(
                {
                    "relevant": retrieval.relevant,
                    "mode": retrieval.mode,
                    "degraded": retrieval.degraded,
                    "top_bm25": retrieval.top_bm25,
                    "top_cosine": retrieval.top_cosine,
                    "hits": [
                        {
                            "citation": h.citation,
                            "score": h.score,
                            "bm25_rank": h.bm25_rank,
                            "bm25_score": h.bm25_score,
                            "vector_rank": h.vector_rank,
                            "cosine": h.cosine,
                        }
                        for h in retrieval.hits
                    ],
                }
                if retrieval
                else None
            ),
            rule_result=result.model_dump(mode="json") if result else None,
            llm=(
                {
                    "provider": self.deps.llm.provider,
                    "model": llm.model,
                    "input_tokens": llm.input_tokens,
                    "output_tokens": llm.output_tokens,
                    "latency_ms": llm.latency_ms,
                    "cost_usd": llm.cost_usd,
                    "prompt_sha256": hashlib.sha256("\n".join(s["prompt"]).encode()).hexdigest(),
                    "draft": llm.text,
                }
                if llm
                else None
            ),
            guardrails=s.get("guardrails", []),
            output_check=(
                {
                    "router": out.router,
                    "decisions": out.model_dump(mode="json")["answers"],
                    "passed": verdict.passed,
                    "reasons": verdict.reasons,
                }
                if out and verdict
                else None
            ),
            answer=s["answer"],
            citations=s.get("citations", []),
            latency_ms=s["timings"],
            cost_usd=cost,
            escalation_reason=s.get("escalation_reason"),
        )
