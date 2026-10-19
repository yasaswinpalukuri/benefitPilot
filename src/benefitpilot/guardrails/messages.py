"""Fixed, reviewed responses for the paths that must never depend on an LLM.

Refusals, emergencies and out-of-scope replies are written once here, so
they can be reviewed like any other code and can never be talked out of
their wording by a prompt.
"""

from __future__ import annotations

from benefitpilot.rules.engine import fmt
from benefitpilot.rules.models import ClaimResult

EMERGENCY = (
    "This sounds like it could be an emergency. Please call 911 or go to the nearest emergency "
    "department now. If you are thinking about suicide or self-harm, you can call or text 988 "
    "(Suicide Crisis Helpline) at any time. I can't help with medical decisions, but I can "
    "explain your coverage once you're safe."
)

REFUSE_MEDICAL = (
    "I can't give medical advice, such as which treatment or medication to use or what a symptom "
    "means. Please speak with a doctor, pharmacist or other health professional; for non-urgent "
    "health advice you can call your provincial health line (811 in most provinces). Once you "
    "know what care you need, I can tell you how your plan covers it."
)

OUT_OF_SCOPE = (
    "I can only help with questions about your health and dental plan: what it covers, your "
    "claims, and how to use your benefits."
)

NOT_FOUND = (
    "I couldn't find that in your plan. I've only got your plan booklet to go on, so for anything "
    "it doesn't cover, please contact your benefits administrator."
)

NEED_MEMBER = (
    "To check a claim I need to know whose coverage to look at. Please select a member and ask "
    "again."
)

NEED_CLAIM_DETAILS = (
    "I can check that for you, but I need a little more detail: which type of service is it "
    "(for example physiotherapy or a dental crown), and what is the amount?"
)


def need_claim_details(service: str | None, amount: str | None) -> str:
    """Ask only for what is missing. Asking "which service?" after the member
    named massage therapy read as if the question had not been understood."""
    if service and not amount:
        return (
            f"I can check that {service.lower()} claim for you, but I couldn't find the "
            "amount. How much was it (for example $120)?"
        )
    if amount and not service:
        return (
            f"I can check that {amount} claim for you, but which type of service is it "
            "(for example physiotherapy or a dental crown)?"
        )
    return NEED_CLAIM_DETAILS


def escalated(audit_id: str) -> str:
    return (
        "I'm not confident I understood your question correctly, so I've passed it to a benefits "
        f"specialist who will follow up. Your reference number is {audit_id[:8]}."
    )


OUTCOME_PHRASES = {
    "approve": "would be covered",
    "partial": "would be partly covered",
    "deny": "would not be covered",
}


def claim_template(result: ClaimResult, label: str) -> str:
    """Deterministic explanation of a rules-engine result. Grounded by construction."""
    parts = [
        f"Based on your plan, this {fmt(result.billed_amount)} {label.lower()} claim "
        f"{OUTCOME_PHRASES[result.outcome]}: the plan would pay {fmt(result.covered_amount)} "
        f"and you would pay {fmt(result.member_pays)}."
    ]
    for c in result.checks:
        if not c.passed:
            cite = f" [{c.citation}]" if c.citation else ""
            parts.append(f"{c.detail}{cite}")
    if result.outcome == "approve" and result.citations:
        parts.append(f"[{result.citations[0]}]")
    return " ".join(parts)
