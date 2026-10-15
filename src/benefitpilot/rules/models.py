"""Inputs and outputs of the rules engine.

Money is `Decimal` end to end. Pydantic serializes Decimal as a JSON string
("80.00"), which keeps cents exact for API clients.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Money = Decimal


class Violation(StrEnum):
    EXCLUDED_SERVICE = "excluded_service"
    NOT_COVERED = "not_covered"
    COVERAGE_NOT_ACTIVE = "coverage_not_active"
    DEPENDENT_INELIGIBLE_AGE = "dependent_ineligible_age"
    WAITING_PERIOD = "waiting_period"
    AGE_LIMIT_EXCEEDED = "age_limit_exceeded"
    FREQUENCY_LIMIT = "frequency_limit"
    TRIP_TOO_LONG = "trip_too_long"
    MISSING_INFORMATION = "missing_information"
    PREAUTH_MISSING = "preauth_missing"
    PER_VISIT_MAX_EXCEEDED = "per_visit_max_exceeded"
    PER_TRIP_MAX_EXCEEDED = "per_trip_max_exceeded"
    ANNUAL_MAX_EXCEEDED = "annual_max_exceeded"
    GROUP_MAX_EXCEEDED = "group_max_exceeded"
    LIFETIME_MAX_EXCEEDED = "lifetime_max_exceeded"


Outcome = Literal["approve", "partial", "deny"]
# deny: the claim is not payable at all. reduce: a limit lowers the payment.
# info: shown for transparency, never a violation (e.g. the deductible).
Effect = Literal["deny", "reduce", "info"]


class ClaimInput(BaseModel):
    """A claim to check. Also the request body of POST /claims/check."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    member_id: str = Field(min_length=1)
    category: str = Field(min_length=1, description="Category or exclusion code from the plan")
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2, description="Billed, CAD")
    service_date: dt.date = Field(alias="date")
    preauth_on_file: bool = False
    trip_days: int | None = Field(default=None, gt=0, description="Required for travel claims")


@dataclass(frozen=True)
class CategoryUsage:
    amount_paid: Money = Decimal("0")  # plan-paid in the benefit year
    last_service_date: dt.date | None = None  # most recent paid service, any year
    lifetime_paid: Money = Decimal("0")


@dataclass(frozen=True)
class MemberSnapshot:
    """Everything the engine needs to know about the patient, as of one benefit year."""

    member_id: str
    plan_id: str
    dob: dt.date
    relationship: Literal["self", "spouse", "child"]
    is_student: bool
    coverage_start: dt.date
    benefit_year: int
    deductible_applied: Money = Decimal("0")
    usage: Mapping[str, CategoryUsage] = field(default_factory=dict)

    def usage_for(self, category: str) -> CategoryUsage:
        return self.usage.get(category, CategoryUsage())


class RuleCheck(BaseModel):
    rule: str
    passed: bool
    effect: Effect
    detail: str
    citation: str | None = None
    violation: Violation | None = None


class ClaimResult(BaseModel):
    member_id: str
    plan_id: str
    category: str
    service_date: dt.date
    outcome: Outcome
    billed_amount: Money
    deductible_applied: Money
    plan_share_before_limits: Money | None
    covered_amount: Money
    member_pays: Money
    violations: list[Violation]
    checks: list[RuleCheck]
    citations: list[str]
