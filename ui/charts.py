"""Charts for the "Why this answer?" panel (pure functions -> Altair charts).

One visual language for every System 1 signal:
- a horizontal bar per option/level/question on a fixed 0-1 axis, one hue
  (each chart is a single series, so no legend), 4px rounded data-ends;
- a thin neutral rule or tick at the decision threshold the policy actually
  uses, labelled, so a reader can see *why* the route was taken;
- value labels in text ink (never the series colour), tooltips with exact values.
Status colours (red/green) are deliberately not used: a refusal isn't "bad data".
"""

from __future__ import annotations

import altair as alt
import pandas as pd

BAR = "#2a78d6"  # reference palette slot 1 (blue); works on light and dark surfaces
RULE = "#8a8984"  # neutral threshold ink, readable on both surfaces
# Value labels use text ink for the active mode (palette text-secondary). Vega's default
# mark text is black, which vanished on the dark theme in the first live check.
INK = {"light": "#52514e", "dark": "#c3c2b7"}
INK_FALLBACK = RULE  # mode unknown (e.g. first load): readable on both


def ink_for(mode: str | None) -> str:
    return INK.get(mode or "", INK_FALLBACK)


BAR_SIZE = 14
# Height per row, not a fixed pixel height: Streamlit overrode the fixed height and
# collapsed every band into one row. A step keeps one row per option however the
# chart is embedded.
ROW_STEP = 28
WIDTH = 360

URGENCY_LEVELS = ["Routine", "Health problem, not urgent", "Possibly life-threatening"]

LABELS = {
    "asks_medical_advice": "Asks for medical advice",
    "needs_member_data": "Needs member's records",
    "answer_contains_medical_advice": "Answer contains medical advice",
    "answer_supported_by_context": "Answer supported by plan text",
    "coverage_question": "Coverage question",
    "claim_check": "Claim check",
    "procedure_how_to": "How-to / procedure",
    "medical_advice_request": "Medical advice request",
    "out_of_scope": "Out of scope",
}


def _bars(df: pd.DataFrame, title: str, ink: str) -> alt.Chart:
    base = alt.Chart(df, title=alt.Title(title, anchor="start", fontSize=13, fontWeight=500))
    y = alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=220))
    bars = base.mark_bar(color=BAR, cornerRadiusEnd=4, size=BAR_SIZE).encode(
        x=alt.X(
            "p:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=True, gridOpacity=0.3),
            title=None,
        ),
        y=y,
        tooltip=[alt.Tooltip("label:N", title="Option"), alt.Tooltip("p:Q", format=".3f")],
    )
    text = base.mark_text(align="left", dx=4, fontSize=11, color=ink).encode(
        x="p:Q", y=y, text=alt.Text("p:Q", format=".0%")
    )
    return bars + text


def _threshold_rule(value: float, caption: str) -> alt.Chart:
    df = pd.DataFrame({"t": [value], "caption": [caption]})
    rule = (
        alt.Chart(df)
        .mark_rule(color=RULE, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="t:Q", tooltip=[alt.Tooltip("caption:N", title="Threshold")])
    )
    label = (
        alt.Chart(df)
        .mark_text(color=RULE, align="left", dx=3, dy=-4, fontSize=10, baseline="bottom")
        .encode(x="t:Q", y=alt.value(0), text="caption:N")
    )
    return rule + label


def intent_chart(answer: dict, auto_route_min: float, ink: str = INK_FALLBACK) -> alt.LayerChart:
    probs = answer["probabilities"]
    df = pd.DataFrame(
        {
            "label": [
                LABELS.get(k, k) + ("  ◀ chosen" if k == answer["choice"] else "") for k in probs
            ],
            "p": list(probs.values()),
        }
    )
    return (
        _bars(df, "Intent (Choice)", ink)
        + _threshold_rule(auto_route_min, f"auto-route ≥ {auto_route_min:.0%}")
    ).properties(width=WIDTH, height=alt.Step(ROW_STEP))


def urgency_chart(answer: dict, emergency_min: float, ink: str = INK_FALLBACK) -> alt.LayerChart:
    probs = {int(k): v for k, v in answer["probabilities"].items()}
    df = pd.DataFrame(
        {
            "label": [URGENCY_LEVELS[i] for i in sorted(probs)],
            "p": [probs[i] for i in sorted(probs)],
        }
    )
    return (
        _bars(df, f"Urgency (Score = {answer['score']:.2f} of 2)", ink)
        + _threshold_rule(emergency_min, f"emergency ≥ {emergency_min:.0%}")
    ).properties(width=WIDTH, height=alt.Step(ROW_STEP))


def noul_chart(
    signals: list[tuple[str, float, float, str]], title: str, ink: str = INK_FALLBACK
) -> alt.LayerChart:
    """signals: (question id, p, threshold, rule text such as 'refuse ≥ 50%')."""
    df = pd.DataFrame(
        {
            "label": [LABELS.get(q, q) for q, _, _, _ in signals],
            "p": [p for _, p, _, _ in signals],
            "t": [t for _, _, t, _ in signals],
            "rule": [r for _, _, _, r in signals],
        }
    )
    y = alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=220))
    ticks = (
        alt.Chart(df)
        .mark_tick(color=RULE, thickness=2, size=BAR_SIZE + 8)
        .encode(x="t:Q", y=y, tooltip=[alt.Tooltip("rule:N", title="Threshold")])
    )
    return (_bars(df, title, ink) + ticks).properties(width=WIDTH, height=alt.Step(ROW_STEP))
