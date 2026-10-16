"""POST /ask, GET /audit/{audit_id}, GET /escalations."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from benefitpilot.api.deps import get_plans
from benefitpilot.audit.log import AuditRecord, AuditStore, Escalation
from benefitpilot.config import get_settings
from benefitpilot.graph.pipeline import Deps, Pipeline
from benefitpilot.llm.client import build_llm
from benefitpilot.retrieval.factory import build_retriever
from benefitpilot.routing.factory import build_router
from benefitpilot.routing.policy import Thresholds
from benefitpilot.store import MemberNotFound, MemberStore

router = APIRouter(tags=["assistant"])


class AskRequest(BaseModel):
    member_id: str | None = Field(default=None, description="Synthetic member ID, e.g. M021")
    question: str = Field(min_length=1, max_length=2000)


class RetrievedChunk(BaseModel):
    citation: str
    title: str
    markdown: str
    score: float
    bm25_rank: int | None
    bm25_score: float | None
    vector_rank: int | None
    cosine: float | None


class AskResponse(BaseModel):
    answer: str
    route: str
    route_reason: str
    citations: list[str]
    rule_result: dict[str, Any] | None
    retrieved: list[RetrievedChunk]  # chunks the answer was grounded in, with scores
    decisions: dict[str, Any]  # raw System 1 probabilities: routing + output check
    guardrails: list[str]
    latency_ms: float
    cost_usd: float
    audit_id: str


@lru_cache
def get_pipeline() -> Pipeline:
    settings = get_settings()
    try:
        store = MemberStore(settings.db_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return Pipeline(
        Deps(
            router=build_router(settings),
            retriever=build_retriever(settings),
            llm=build_llm(settings),
            store=store,
            plans=get_plans(),
            thresholds=Thresholds.load(settings.router_backend),
            audit=AuditStore(settings.audit_db_path),
        )
    )


@lru_cache
def get_audit() -> AuditStore:
    return AuditStore(get_settings().audit_db_path)


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, pipeline: Annotated[Pipeline, Depends(get_pipeline)]) -> AskResponse:
    try:
        state, record = await pipeline.ask(req.question, req.member_id)
    except MemberNotFound as e:
        raise HTTPException(status_code=404, detail=f"Unknown member {req.member_id}") from e
    decisions = {"routing": record.router["decisions"]}
    if record.output_check:
        decisions["output_check"] = record.output_check["decisions"]
    retrieval = state.get("retrieval")
    retrieved = [
        RetrievedChunk(
            citation=h.citation,
            title=h.chunk.title,
            markdown=h.chunk.markdown,
            score=h.score,
            bm25_rank=h.bm25_rank,
            bm25_score=h.bm25_score,
            vector_rank=h.vector_rank,
            cosine=h.cosine,
        )
        for h in (retrieval.hits if retrieval else [])
    ]
    return AskResponse(
        answer=record.answer,
        route=record.route,
        route_reason=record.route_reason,
        citations=record.citations,
        rule_result=record.rule_result,
        retrieved=retrieved,
        decisions=decisions,
        guardrails=record.guardrails,
        latency_ms=record.latency_ms["total"],
        cost_usd=record.cost_usd,
        audit_id=record.audit_id,
    )


@router.get("/audit/{audit_id}", response_model=AuditRecord)
def audit(audit_id: str, store: Annotated[AuditStore, Depends(get_audit)]) -> AuditRecord:
    record = store.get(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown audit_id")
    return record


@router.get("/escalations", response_model=list[Escalation])
def escalations(
    store: Annotated[AuditStore, Depends(get_audit)], status: str = "open"
) -> list[Escalation]:
    return store.escalations(status)
