"""Fill the rules engine's arguments from a natural-language claim question.

"I've used $420 of physio. Will a $120 session be covered?" needs a category
and an amount. Following TypeSafe's "select instead of generate" pattern:

- code finds every candidate dollar amount (with its surrounding words);
- System 1 *selects* the category (from the member's plan) and the amount
  (from the candidates). Nothing is generated, so the engine can only ever
  receive a real plan category and an amount the member actually typed.

The two questions are speculative: they go out in the same request as the
routing questions and are used only if the route is `claim_check`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from benefitpilot.routing.types import Decisions, Question
from benefitpilot.rules.plan_schema import Plan

CLAIM_CATEGORY = "claim_category"
CLAIM_AMOUNT = "claim_amount"
UNCLEAR = "unclear"
MIN_SELECTION_P = 0.5

_NUMBER = r"\d[\d,]*(?:\.\d{1,2})?"
# A number counts as money only with a currency marker before or after it: "$450",
# "CAD 450", "450 dollars", "450$", "450 bucks". Only "$" was recognised at first,
# so "it cost around 450 dollars" reached the engine with no amount (seen live).
# Bare numbers stay out: "2 sessions" or "24 months" are not costs.
AMOUNT_RE = re.compile(
    rf"(?:\$|\b(?:CAD|USD)\s?)\s?(?P<pre>{_NUMBER})"
    rf"|(?<![\d.,$])(?P<post>{_NUMBER})\s?(?:\$|dollars?\b|bucks\b|CAD\b|USD\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AmountCandidate:
    key: str  # "amount_1"
    amount: Decimal
    context: str  # the words around it, shown to the model


def find_amounts(text: str, window: int = 40) -> list[AmountCandidate]:
    out = []
    for i, m in enumerate(AMOUNT_RE.finditer(text), start=1):
        start, end = max(0, m.start() - window), min(len(text), m.end() + window)
        out.append(
            AmountCandidate(
                key=f"amount_{i}",
                amount=Decimal((m.group("pre") or m.group("post")).replace(",", "")),
                context=text[start:end].strip(),
            )
        )
    return out


def category_question(plan: Plan) -> Question:
    options = {name: cat.label for name, cat in plan.categories.items()}
    options |= {x.code: f"{x.label} (excluded from the plan)" for x in plan.exclusions.items}
    options[UNCLEAR] = "The message does not say which service, or it is not any of these."
    return Question(
        id=CLAIM_CATEGORY,
        kind="choice",
        instructions="If `member_message` asks whether a specific expense will be covered, "
        "which type of service is that expense for?",
        criteria=options,
    )


def amount_question(candidates: list[AmountCandidate]) -> Question:
    options: dict[str, str] = {c.key: f"${c.amount} — “{c.context}”" for c in candidates}
    options["none"] = "None of these is the cost of the expense being asked about."
    return Question(
        id=CLAIM_AMOUNT,
        kind="choice",
        instructions="`member_message` may mention several dollar amounts, such as money already "
        "spent this year and the cost of a new expense. Which amount is the cost of the new "
        "expense the member is asking about?",
        criteria=options,
    )


def claim_questions(plan: Plan, message: str) -> list[Question]:
    questions = [category_question(plan)]
    candidates = find_amounts(message)
    if candidates:
        questions.append(amount_question(candidates))
    return questions


@dataclass(frozen=True)
class ClaimArgs:
    category: str | None
    amount: Decimal | None
    category_p: float
    amount_p: float

    @property
    def complete(self) -> bool:
        return self.category is not None and self.amount is not None


def read_claim_args(decisions: Decisions, message: str) -> ClaimArgs:
    category, category_p = None, 0.0
    if CLAIM_CATEGORY in decisions.answers:
        c = decisions.choice(CLAIM_CATEGORY)
        category_p = c.top_p
        if c.choice != UNCLEAR and c.top_p >= MIN_SELECTION_P:
            category = c.choice
    amount, amount_p = None, 0.0
    if CLAIM_AMOUNT in decisions.answers:
        a = decisions.choice(CLAIM_AMOUNT)
        amount_p = a.top_p
        by_key = {c.key: c.amount for c in find_amounts(message)}
        if a.choice in by_key and a.top_p >= MIN_SELECTION_P:
            amount = by_key[a.choice]
    return ClaimArgs(category, amount, round(category_p, 4), round(amount_p, 4))
