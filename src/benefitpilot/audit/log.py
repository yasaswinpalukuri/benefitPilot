"""Structured audit log and human-escalation queue (SQLite).

Every /ask request writes one `AuditRecord`: the question, the raw System 1
probabilities, the route and why, the retrieved chunks with their scores, the
rules-engine result, the LLM call (model, tokens, cost, prompt hash), the
output check, the final answer, per-stage latency and total cost.

Raw probabilities are stored so thresholds can be re-tuned later by replaying
the log, with no new inference (CLAUDE.md, non-negotiable 5). The record is
stored as JSON next to a few indexed columns for querying.

This database is separate from the member database: the member store is opened
read-only, and audit writes never touch it.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    audit_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    member_id TEXT,
    route TEXT NOT NULL,
    router TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_created ON audit(created_at);
CREATE TABLE IF NOT EXISTS escalations (
    audit_id TEXT PRIMARY KEY REFERENCES audit(audit_id),
    created_at TEXT NOT NULL,
    member_id TEXT,
    question TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);
"""


def new_audit_id() -> str:
    return uuid.uuid4().hex


class AuditRecord(BaseModel):
    audit_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    member_id: str | None
    question: str
    route: str
    route_reason: str
    router: dict[str, Any]  # name, model, raw decisions, latency, tokens, cost
    claim_args: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    rule_result: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    guardrails: list[str] = Field(default_factory=list)  # what fired and why
    output_check: dict[str, Any] | None = None
    answer: str
    citations: list[str]
    latency_ms: dict[str, float]  # per stage, plus "total"
    cost_usd: float
    escalation_reason: str | None = None


class Escalation(BaseModel):
    audit_id: str
    created_at: str
    member_id: str | None
    question: str
    reason: str
    status: str


class AuditStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def write(self, record: AuditRecord) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO audit VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.audit_id,
                    record.created_at,
                    record.member_id,
                    record.route,
                    record.router.get("name", ""),
                    record.model_dump_json(),
                ),
            )
            if record.escalation_reason:
                con.execute(
                    "INSERT INTO escalations (audit_id, created_at, member_id, question, reason)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        record.audit_id,
                        record.created_at,
                        record.member_id,
                        record.question,
                        record.escalation_reason,
                    ),
                )

    def get(self, audit_id: str) -> AuditRecord | None:
        with self._connect() as con:
            row = con.execute("SELECT record FROM audit WHERE audit_id = ?", (audit_id,)).fetchone()
        return AuditRecord.model_validate_json(row["record"]) if row else None

    def escalations(self, status: str = "open") -> list[Escalation]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM escalations WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        return [Escalation(**dict(r)) for r in rows]
