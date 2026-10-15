"""Typed schema for the machine-readable plan rules in `config/plans/*.yaml`.

Conventions (also stated in docs/DECISIONS.md, ADR-005):
- All money is CAD. Annual/lifetime maximums cap what the *plan pays*
  (the reimbursed amount), not the amount billed.
- `coinsurance` is the fraction the plan pays (0.8 = plan pays 80%).
- Age limits are exclusive: `age_limit: 19` means "under age 19".
- `None` means "no limit of this kind".
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benefitpilot.config import PLAN_RULES_DIR


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Frequency(_Strict):
    # Only "1 per N months" is supported: usage stores the last service date,
    # not a full service history (docs/DECISIONS.md, ADR-012).
    count: Literal[1]
    months: int = Field(gt=0)


class Category(_Strict):
    section: str
    label: str
    group: str | None = None
    coinsurance: float = Field(gt=0, le=1)
    deductible_applies: bool
    annual_max: float | None = None
    per_visit_max: float | None = None
    frequency: Frequency | None = None
    age_limit: int | None = None
    lifetime_max: float | None = None
    per_trip_max: float | None = None
    max_trip_days: int | None = None


class Group(_Strict):
    section: str
    label: str
    combined_annual_max: float | None = None


class Eligibility(_Strict):
    section: str
    child_age_limit: int
    student_age_limit: int


class Deductible(_Strict):
    section: str
    annual_per_person: float = Field(ge=0)


class WaitingPeriods(_Strict):
    section: str
    months: dict[str, int]


class Preauthorization(_Strict):
    section: str
    # 0 means every claim in the category needs pre-authorization;
    # otherwise claims strictly over this amount need it.
    thresholds: dict[str, float]


class Exclusion(_Strict):
    code: str
    label: str


class Exclusions(_Strict):
    section: str
    items: list[Exclusion]


class Plan(_Strict):
    plan_id: str
    name: str
    benefit_year: str
    currency: str
    eligibility: Eligibility
    deductible: Deductible
    groups: dict[str, Group]
    categories: dict[str, Category]
    waiting_periods: WaitingPeriods
    preauthorization: Preauthorization
    exclusions: Exclusions

    @model_validator(mode="after")
    def _check_references(self) -> Plan:
        cats = set(self.categories)
        for name, cat in self.categories.items():
            if cat.group is not None and cat.group not in self.groups:
                raise ValueError(f"{name}: unknown group {cat.group!r}")
        for field, keys in (
            ("waiting_periods", self.waiting_periods.months),
            ("preauthorization", self.preauthorization.thresholds),
        ):
            unknown = set(keys) - cats
            if unknown:
                raise ValueError(f"{field} references unknown categories: {sorted(unknown)}")
        overlap = cats & {e.code for e in self.exclusions.items}
        if overlap:
            raise ValueError(f"categories cannot also be excluded: {sorted(overlap)}")
        sections = [c.section for c in self.categories.values()] + [
            g.section for g in self.groups.values()
        ]
        if len(sections) != len(set(sections)):
            raise ValueError("section IDs must be unique")
        return self

    def citation(self, section: str) -> str:
        """Stable citation ID, e.g. 'PLAN-A §4.2.1'."""
        return f"{self.plan_id} §{section}"


def load_plan(path: Path) -> Plan:
    with path.open(encoding="utf-8") as f:
        return Plan.model_validate(yaml.safe_load(f))


def load_plans(directory: Path = PLAN_RULES_DIR) -> dict[str, Plan]:
    plans = [load_plan(p) for p in sorted(directory.glob("*.yaml"))]
    return {p.plan_id: p for p in plans}
