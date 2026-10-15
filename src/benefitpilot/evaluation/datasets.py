"""Golden set and red-team set.

Both are split 50/50 into `dev` (used to tune thresholds) and `test` (used for
the headline numbers), so no reported number comes from data that was used
to choose a threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from benefitpilot.config import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval"
GOLDEN = EVAL_DIR / "golden_set.jsonl"
REDTEAM = EVAL_DIR / "redteam.jsonl"
SAFE_ROUTES = {"refuse_medical", "emergency"}


@dataclass(frozen=True)
class EvalItem:
    id: str
    split: Literal["dev", "test"]
    question: str
    member_id: str | None
    route: str  # expected route
    intent: str | None = None  # golden only
    medical: int | None = None  # 1 = asks for medical advice
    urgency: int | None = None
    member_data: int | None = None
    citations: list[str] = field(default_factory=list)
    source: Literal["golden", "redteam"] = "golden"
    style: str | None = None  # red-team attack style

    @property
    def is_guardrail_case(self) -> bool:
        """Must be refused or treated as an emergency."""
        return self.route in SAFE_ROUTES


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_golden(path: Path = GOLDEN) -> list[EvalItem]:
    return [EvalItem(**row, source="golden") for row in _read(path)]


def load_redteam(path: Path = REDTEAM) -> list[EvalItem]:
    return [
        EvalItem(
            id=row["id"],
            split=row["split"],
            question=row["question"],
            member_id=row.get("member_id"),
            route=row["expected"],
            medical=1,
            urgency=2 if row["expected"] == "emergency" else None,
            source="redteam",
            style=row.get("style"),
        )
        for row in _read(path)
    ]


def load_all() -> list[EvalItem]:
    return load_golden() + load_redteam()
