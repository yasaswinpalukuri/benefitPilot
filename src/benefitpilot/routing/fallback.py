"""FallbackRouter: a small local classifier behind the same interface as Jev.

It lets the project run with no API key and gives the benchmark a baseline.

- intent / asks_medical_advice / urgency / needs_member_data: TF-IDF (word
  1-2 grams plus character 3-5 grams, so "physio" ~ "physiotherapy") and
  logistic regression, trained at startup from `data/router_training.jsonl`
  in well under a second. There is no pickled model to ship or trust.
- Safety layers: medical advice and emergencies also go through keyword
  rules tuned for *recall*, and the final probability is the max of the model
  and the rule. A false refusal is cheap; a missed medical-advice request or
  emergency is not (CLAUDE.md §4).
- Output check: deterministic heuristics (advice phrasing; numbers in the
  answer that don't appear in the retrieved text). This is a deliberately weak
  baseline; Jev is the intended output checker.
"""

from __future__ import annotations

import json
import re
import time
from importlib import resources

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from benefitpilot.guardrails.grounding import unsupported_numbers
from benefitpilot.routing.confidence import choice_confidence, score_confidence
from benefitpilot.routing.questions import INTENTS
from benefitpilot.routing.types import (
    ChoiceDecision,
    Decisions,
    NoulDecision,
    Question,
    RouterError,
    RouterState,
    ScoreDecision,
)

SYMPTOM = (
    r"pain|hurts?|ache|aching|sore|fever|cough|rash|dizz\w*|nause\w*|vomit\w*|bleed\w*|"
    r"infection|swell\w*|swollen|migraine|headache|blood pressure|sprain\w*|fracture|"
    r"throbbing|numb|allerg\w*|insomnia|can'?t sleep|lump|sciatica"
)
TREATMENT = (
    r"medication|medicine|meds|pills?|drugs?|dose|dosage|mg\b|ibuprofen|advil|tylenol|"
    r"acetaminophen|aspirin|antibiotics?|antidepressants?|melatonin|painkillers?|"
    r"stretch\w*|exercises?|treatment|surgery|ice|heat"
)
ADVICE = (
    r"should i|what (should|can|do) i|which .* (best|better)|(best|better) .* for|"
    r"is it safe|how (much|many|often) .* take|what could it be|is (this|that|it) "
    r"(serious|bad|normal)|do i need|can i stop|tell me what to|recommend|what to take|"
    r"how do i (make|stop|treat|cure)|pretend you'?re (my|a) doctor|what dose"
)
EMERGENCY = (
    r"chest pain|crushing|can'?t breathe|trouble breathing|turning blue|overdos\w*|"
    r"whole bottle|suicid\w*|kill(ing)? myself|end my life|self[- ]harm|collapsed|"
    r"not responding|unconscious|won'?t stop bleeding|severe bleeding|throat is swelling|"
    r"face is drooping|stroke|heart attack|seizure"
)
ANSWER_ADVICE = (
    r"\b(you should|i recommend|i'd recommend|try taking|take \d|take (some|an?)\b)|"
    r"\b\d+\s?mg\b|\b(ibuprofen|acetaminophen|tylenol|advil|antibiotic)\b"
)
DURATION = r"\b\d+\s+(?:months?|days?)\b"


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


SYMPTOM_RX, TREATMENT_RX, ADVICE_RX = _rx(SYMPTOM), _rx(TREATMENT), _rx(ADVICE)
EMERGENCY_RX, ANSWER_ADVICE_RX, DURATION_RX = _rx(EMERGENCY), _rx(ANSWER_ADVICE), _rx(DURATION)


def medical_rule(text: str) -> float:
    """High-recall rule: advice-seeking phrasing about a symptom or treatment."""
    if ADVICE_RX.search(text) and (SYMPTOM_RX.search(text) or TREATMENT_RX.search(text)):
        return 0.9
    return 0.0


def emergency_rule(text: str) -> float:
    return 0.9 if EMERGENCY_RX.search(text) else 0.0


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "physiotherapy": ("physio",),
    "chiropractic": ("chiro",),
    "massage_therapy": ("massage",),
    "psychology": ("psycholog", "therap", "counsel"),
    "eye_exam": ("eye exam", "optometrist", "ophthalmologist"),
    "eyewear": ("glasses", "contacts", "contact lens", "eyewear", "frames"),
    "travel_emergency": ("travel", "abroad", "trip", "out of province"),
    "dental_basic": ("cleaning", "filling", "checkup", "x-ray", "extraction"),
    "dental_major": ("crown", "bridge", "denture", "root canal"),
    "orthodontics": ("braces", "orthodont", "aligner", "invisalign"),
    "cosmetic_dental": ("whitening", "cosmetic"),
    "laser_eye_surgery": ("laser", "lasik"),
    "naturopathy": ("naturopath",),
    "experimental_treatment": ("experimental",),
    "drug_tier_1": ("generic",),
}
COST_WORDS = re.compile(
    r"session|visit|bill|claim|cost|charge|receipt|appointment|pair|"
    r"crown|treatment|will|would|next|new",
    re.IGNORECASE,
)
SPENT_WORDS = re.compile(r"used|spent|already|so far|this year|claimed", re.IGNORECASE)


def _pick(options: list[str], chosen: str, p: float = 0.9) -> ChoiceDecision:
    if chosen not in options:
        chosen = options[-1]
    rest = (1 - p) / (len(options) - 1) if len(options) > 1 else 0.0
    probs = {o: (p if o == chosen else rest) for o in options}
    if len(options) == 1:
        probs[chosen] = 1.0
    return ChoiceDecision(
        choice=chosen, probabilities=probs, confidence=choice_confidence(list(probs.values()))
    )


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ]
    )


def _classifier() -> Pipeline:
    return Pipeline(
        [
            ("features", _features()),
            ("lr", LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")),
        ]
    )


def load_training_examples() -> list[dict]:
    text = resources.files("benefitpilot.routing").joinpath("data/router_training.jsonl")
    return [json.loads(line) for line in text.read_text(encoding="utf-8").splitlines() if line]


class FallbackRouter:
    name = "fallback"

    def __init__(self, examples: list[dict] | None = None):
        examples = examples or load_training_examples()
        texts = [e["text"] for e in examples]
        self.models: dict[str, Pipeline] = {}
        for target in ("intent", "medical", "urgency", "member_data"):
            model = _classifier()
            model.fit(texts, [e[target] for e in examples])
            self.models[target] = model
        missing = set(INTENTS) - set(self.models["intent"].classes_)
        if missing:
            raise ValueError(f"training data has no examples for intents: {sorted(missing)}")
        self._handlers = {
            "intent": self._intent,
            "asks_medical_advice": self._medical,
            "urgency": self._urgency,
            "needs_member_data": self._member_data,
            "answer_contains_medical_advice": self._answer_medical,
            "answer_supported_by_context": self._answer_supported,
            "claim_category": self._claim_category,
            "claim_amount": self._claim_amount,
        }

    def _proba(self, target: str, text: str) -> dict:
        model = self.models[target]
        return dict(zip(model.classes_, model.predict_proba([text])[0], strict=True))

    # --- routing questions ---------------------------------------------------

    def _intent(self, state: RouterState, q: Question) -> ChoiceDecision:
        proba = self._proba("intent", state.member_message)
        probs = {opt: float(proba.get(opt, 0.0)) for opt in q.options}
        choice = max(probs, key=probs.__getitem__)
        return ChoiceDecision(
            choice=choice,
            probabilities=probs,
            confidence=choice_confidence(list(probs.values())),
        )

    def _medical(self, state: RouterState, q: Question) -> NoulDecision:
        model_p = float(self._proba("medical", state.member_message).get(1, 0.0))
        return NoulDecision(p=max(model_p, medical_rule(state.member_message)))

    def _urgency(self, state: RouterState, q: Question) -> ScoreDecision:
        proba = self._proba("urgency", state.member_message)
        p = np.array([float(proba.get(level, 0.0)) for level in range(len(q.criteria))])
        top = len(p) - 1
        rule = emergency_rule(state.member_message)
        if rule > p[top]:
            # Raise the top level to the rule's floor and rescale the rest to sum to 1.
            rest = p[:top].sum()
            p[:top] = p[:top] * (1 - rule) / rest if rest else (1 - rule) / top
            p[top] = rule
        probs = {i: float(v) for i, v in enumerate(p)}
        return ScoreDecision(
            score=float(sum(i * v for i, v in probs.items())),
            probabilities=probs,
            confidence=score_confidence(list(p)),
        )

    def _member_data(self, state: RouterState, q: Question) -> NoulDecision:
        return NoulDecision(p=float(self._proba("member_data", state.member_message).get(1, 0.0)))

    # --- claim arguments (keyword baseline) -----------------------------------

    def _claim_category(self, state: RouterState, q: Question) -> ChoiceDecision:
        text = state.member_message.lower()
        hits = [opt for opt in q.options if any(k in text for k in CATEGORY_KEYWORDS.get(opt, ()))]
        return _pick(q.options, hits[0] if len(hits) == 1 else "unclear")

    def _claim_amount(self, state: RouterState, q: Question) -> ChoiceDecision:
        """Prefer an amount near a cost word and away from words about past spending."""
        texts = q.criteria
        scored = []
        for opt in q.options:
            if opt == "none":
                continue
            context = str(texts[opt]).lower()
            score = len(COST_WORDS.findall(context)) - 2 * len(SPENT_WORDS.findall(context))
            scored.append((score, opt))
        if not scored:
            return _pick(q.options, "none")
        best = max(scored)  # ties go to the later amount (usually the new expense)
        ties = [o for s, o in scored if s == best[0]]
        return _pick(q.options, ties[-1])

    # --- output check --------------------------------------------------------

    def _answer_medical(self, state: RouterState, q: Question) -> NoulDecision:
        return NoulDecision(
            p=0.9 if ANSWER_ADVICE_RX.search(state.assistant_answer or "") else 0.05
        )

    def _answer_supported(self, state: RouterState, q: Question) -> NoulDecision:
        answer, context = state.assistant_answer or "", state.retrieved_plan_text or ""
        if not context:
            return NoulDecision(p=0.05)
        # Compare values, not strings: "$80" in an answer matches "$80.00" in the context.
        if unsupported_numbers(answer, context) or self._unsupported_durations(answer, context):
            return NoulDecision(p=0.1)  # a figure the plan text doesn't contain
        return NoulDecision(p=0.8)

    @staticmethod
    def _unsupported_durations(answer: str, context: str) -> set[str]:
        def norm(text: str) -> set[str]:
            found = DURATION_RX.findall(text)
            return {re.sub(r"s$", "", d.lower().replace(" ", "")) for d in found}

        return norm(answer) - norm(context)

    async def decide(self, state: RouterState, questions: list[Question]) -> Decisions:
        start = time.perf_counter()
        answers = {}
        for q in questions:
            handler = self._handlers.get(q.id)
            if handler is None:
                raise RouterError(f"FallbackRouter cannot answer question {q.id!r}")
            answers[q.id] = handler(state, q)
        return Decisions(
            answers=answers,
            router=self.name,
            model="tfidf-logreg+rules",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
