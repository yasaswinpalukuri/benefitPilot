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

# Lexicons are general clinical vocabulary and standard triage red flags, written
# from domain knowledge, not from eval phrasings (ADR-028). Broadened in Phase 7
# after the dev-split red-team run showed 56% recall with the Phase 4 lists.
SYMPTOM = (
    r"pain|hurts?|ache|aching|sore|fever|cough\w*|rash|dizz\w*|nause\w*|vomit\w*|bleed\w*|"
    r"infection|swell\w*|swollen|migraine|headache|sprain\w*|fracture|broken|throbbing|"
    r"numb|allerg\w*|insomnia|can'?t sleep|lump|sciatica|diarrh\w*|constipat\w*|"
    r"tired|fatigue|itch\w*|burn\w*|cramp\w*|stiff|weak\w*|faint\w*|mole|spots?|"
    r"symptoms?|side effects?"
)
CONDITION = (
    r"diabet\w*|blood (sugar|pressure)|cholesterol|depress\w*|anxiety|panic|ptsd|adhd|"
    r"asthma|eczema|arthritis|cancer|tumou?r|thyroid|pregnan\w*|uti|flu|covid|cold|"
    r"strep|ear infection|herniat\w*|disc|acl|concussion|insulin|stitches|wound|"
    r"vaccin\w*|immuni[sz]\w*|vitamin \w+|supplements?|diet|weight loss"
)
TREATMENT = (
    r"medications?|medicines?|meds|pills?|tablets?|drugs?|doses?|dosage|dosing|mg\b|"
    r"prescriptions?|over[- ]the[- ]counter|otc|ibuprofen|advil|motrin|tylenol|"
    r"acetaminophen|aspirin|naproxen|antibiotics?|antidepressants?|ssris?|zoloft|prozac|"
    r"xanax|oxycodone|opioids?|melatonin|painkillers?|sleeping pills?|blood thinners?|"
    r"chemo\w*|epipen|inhaler|stretch\w*|exercises?|treatments?|therapy|surgery|"
    r"\bice\b|\bheat\b|rest\b"
)
ADVICE = (
    r"should i|what (should|can|do) i|which .*(best|better|work)|(best|better|fastest) .*for|"
    r"is it (safe|okay|ok|bad|normal|serious)|is (this|that) (serious|bad|normal)|"
    # "how much/many/often" only with an advice verb: "How much does the plan pay for
    # massage therapy?" is a plan question, "how much ibuprofen can I take" is not.
    r"how (much|many|often|long)\b.*\b(take|use|drink|eat|need|wait|give|safe)|"
    r"should (my|i|we)\b|what .*\b(stops?|cures?|treats?|relieves?)\b|"
    r"what could it be|what does it mean|do i have\b|"
    r"should i (worry|be worried)|am i\b|do i need|can i (stop|skip|double|mix|give|use|"
    r"take|drink)|tell me what to|recommend|what to take|right amount|safe (amount|dose)|"
    r"how do i (make|stop|treat|cure|set|fix)|pretend you'?re (my|a) doctor|what dose|"
    r"diagnos\w*|guide to (treating|setting)|treat (my|their|his|her)|what happens now|"
    r"what('s| is) (the |a )?(max|maximum|lethal)|helps? with|would help|good for"
)
EMERGENCY = (
    r"chest (pain|pressure|tightness|feels tight)|crushing|can'?t breathe|"
    r"trouble breathing|short(ness)? of breath|turning blue|overdos\w*|whole bottle|"
    r"too many (pills|tablets|sleeping pills)|swallowed (a |an |some )?(battery|button "
    r"battery|bleach|poison|magnet|pills)|poison\w*|suicid\w*|kill(ing)? myself|"
    r"end (my life|it all)|(hurt|harm)(ing)? myself|self[- ]harm|don'?t want to (live|be "
    r"alive)|collapsed|not responding|unconscious|passed out|won'?t stop bleeding|"
    r"severe bleeding|(throat|tongue|lips?|face) (is |are )?swell\w*|face is drooping|"
    r"slurred speech|stroke|heart attack|seizure|stiff neck"
)
ANSWER_ADVICE = (
    r"\b(you should|i recommend|i'd recommend|try taking|take \d|take (some|an?)\b)|"
    r"\b\d+\s?mg\b|\b(ibuprofen|acetaminophen|tylenol|advil|antibiotic)\b"
)
DURATION = r"\b\d+\s+(?:months?|days?)\b"


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


SYMPTOM_RX, TREATMENT_RX, ADVICE_RX = _rx(SYMPTOM), _rx(TREATMENT), _rx(ADVICE)
CONDITION_RX = _rx(CONDITION)
EMERGENCY_RX, ANSWER_ADVICE_RX, DURATION_RX = _rx(EMERGENCY), _rx(ANSWER_ADVICE), _rx(DURATION)


def medical_rule(text: str) -> float:
    """High-recall rule: advice-seeking phrasing about a symptom or treatment."""
    clinical = SYMPTOM_RX.search(text) or CONDITION_RX.search(text) or TREATMENT_RX.search(text)
    if ADVICE_RX.search(text) and clinical:
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
