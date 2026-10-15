"""Deterministic grounding checks on LLM output.

These run on every answer, before the System 1 output check:
- money: every dollar amount and percentage in a claim explanation must come
  from the rules engine (non-negotiable 4: money is computed by code);
- citations: only section IDs that were actually given to the model count.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")
CITATION_RE = re.compile(r"\[?(PLAN-[A-Z0-9]+ §[\d.]*\d)\]?")


def _amounts(text: str) -> set[Decimal]:
    out = set()
    for m in MONEY_RE.findall(text):
        try:
            out.add(Decimal(m.replace(",", "")).quantize(Decimal("0.01")))
        except InvalidOperation:
            continue
    return out


def _percents(text: str) -> set[Decimal]:
    return {Decimal(p).normalize() for p in PERCENT_RE.findall(text)}


def unsupported_numbers(answer: str, source: str) -> list[str]:
    """Dollar amounts and percentages in `answer` that don't appear in `source`."""
    missing = [f"${a}" for a in sorted(_amounts(answer) - _amounts(source))]
    missing += [f"{p}%" for p in sorted(_percents(answer) - _percents(source))]
    return missing


NEGATIVE_RE = re.compile(
    r"(?:\bnot|n't|\bnever)\s+(?:be\s+|been\s+)?(?:covered|paid|reimbursed|eligible|payable)\b"
    r"|\bden(y|ied|ial)\b|\bdecline[ds]?\b|\breject(ed)?\b",
    re.IGNORECASE,
)


def claim_inconsistencies(
    answer: str, outcome: str, covered: Decimal, member_pays: Decimal
) -> list[str]:
    """Problems with a claim explanation that a number check alone can't see.

    A draft can repeat only real numbers and still reverse the conclusion
    ("$80 already paid ... a new session would not be covered" when the engine
    says $80 *will* be paid). So:
    - polarity: a covered/partly covered result must not be described as
      denied, and a denial must say so;
    - both bottom-line amounts must be stated.
    """
    problems = []
    negative = bool(NEGATIVE_RE.search(answer))
    if outcome in ("approve", "partial") and negative:
        problems.append(f"says not covered, but outcome is {outcome}")
    if outcome == "deny" and not negative:
        problems.append("does not say the claim is not covered")
    amounts = _amounts(answer)
    for label, value in (("plan pays", covered), ("member pays", member_pays)):
        if value.quantize(Decimal("0.01")) not in amounts:
            problems.append(f"does not state what the {label} ({value})")
    return problems


def cited_ids(answer: str, allowed: list[str]) -> list[str]:
    """Citations in `answer` that are in `allowed`, in order of first appearance."""
    seen: list[str] = []
    for c in CITATION_RE.findall(answer):
        if c in allowed and c not in seen:
            seen.append(c)
    return seen
