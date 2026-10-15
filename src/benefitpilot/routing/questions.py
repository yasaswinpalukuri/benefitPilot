"""The typed questions BenefitPilot asks System One.

Design rules from the TypeSafe docs: one narrow judgment per question; a
no-match option (`out_of_scope`); Score levels describe concrete situations,
not degrees; Noul phrased so that yes = the thing we act on. Routing questions
are independent, so they go out together in one request; the output check
needs the LLM's answer, so it is a second request.
"""

from __future__ import annotations

from benefitpilot.routing.types import Question

INTENTS = (
    "coverage_question",
    "claim_check",
    "procedure_how_to",
    "medical_advice_request",
    "out_of_scope",
)

INTENT = Question(
    id="intent",
    kind="choice",
    instructions=(
        "A member of a health and dental insurance plan sent `member_message`. "
        "What kind of help are they asking for?"
    ),
    criteria={
        "coverage_question": {
            "what": "What the plan covers in general: benefits, limits, maximums, "
            "coinsurance, deductibles, waiting periods, exclusions, eligibility.",
            "not_for": "Questions about a specific claim amount or the member's own "
            "remaining balance (claim_check).",
        },
        "claim_check": {
            "what": "Whether a specific expense or claim for this member will be paid, "
            "how much they have used or have left, or why a claim was paid a certain amount.",
            "examples": [
                "I've used $420 of physio. Will a $120 session be covered?",
                "How much massage do I have left this year?",
            ],
        },
        "procedure_how_to": {
            "what": "How to do something with the plan: submit a claim, get "
            "pre-authorization, coordinate with a spouse's plan, appeal a decision.",
        },
        "medical_advice_request": {
            "what": "Asks what is clinically right for them: which treatment, medication "
            "or dose to use, what a symptom means, or whether they need care.",
            "not_for": "Asking whether a treatment is covered by the plan.",
        },
        "out_of_scope": {
            "what": "Anything not about this health plan or the member's health, "
            "or a request the assistant should not handle.",
        },
    },
)

ASKS_MEDICAL_ADVICE = Question(
    id="asks_medical_advice",
    kind="noul",
    instructions=(
        "Does `member_message` ask for medical advice: what treatment, medication or "
        "dose they should use, what a symptom means, or whether they need care?"
    ),
    criteria={
        "true": "The member wants a clinical recommendation or interpretation, even if "
        "they also ask about coverage.",
        "false": "The member only asks what the plan covers or how the plan works, "
        "for example whether physiotherapy or a drug is covered.",
    },
)

URGENCY = Question(
    id="urgency",
    kind="score",
    instructions="How urgent is the member's health situation described in `member_message`?",
    criteria=[
        "No health problem is happening now; the member asks about benefits, claims or paperwork.",
        "The member mentions a current health problem or symptom that is not life-threatening, "
        "such as ongoing back pain or a toothache.",
        "The member describes a possibly life-threatening situation happening now, such as "
        "chest pain, trouble breathing, severe bleeding, an overdose, or thoughts of suicide "
        "or self-harm.",
    ],
)

NEEDS_MEMBER_DATA = Question(
    id="needs_member_data",
    kind="noul",
    instructions=(
        "To answer `member_message`, do we need this member's own records, such as how "
        "much they have already claimed this year or their coverage start date?"
    ),
    criteria={
        "true": "The answer depends on the member's personal usage, balances or dates.",
        "false": "The plan booklet alone answers it.",
    },
)

ROUTING_QUESTIONS = [INTENT, ASKS_MEDICAL_ADVICE, URGENCY, NEEDS_MEMBER_DATA]

ANSWER_CONTAINS_MEDICAL_ADVICE = Question(
    id="answer_contains_medical_advice",
    kind="noul",
    instructions=(
        "Does `assistant_answer` give medical advice: recommend a treatment, medication or "
        "dose, interpret a symptom, or tell the member whether they need care?"
    ),
    criteria={
        "true": "It tells the member what to do clinically, beyond saying what the plan covers "
        "or suggesting they consult a health professional or emergency services.",
        "false": "It only describes coverage, amounts, or plan procedures, or refers the member "
        "to a health professional.",
    },
)

ANSWER_SUPPORTED_BY_CONTEXT = Question(
    id="answer_supported_by_context",
    kind="noul",
    instructions=(
        "Is every statement about coverage, amounts, limits or procedures in `assistant_answer` "
        "supported by `retrieved_plan_text`?"
    ),
    criteria={
        "true": "Each such statement can be found in, or directly follows from, "
        "`retrieved_plan_text`.",
        "false": "At least one statement is missing from or contradicts `retrieved_plan_text`.",
    },
)

OUTPUT_CHECK_QUESTIONS = [ANSWER_CONTAINS_MEDICAL_ADVICE, ANSWER_SUPPORTED_BY_CONTEXT]
