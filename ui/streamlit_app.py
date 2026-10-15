"""BenefitPilot demo UI (Streamlit).

    streamlit run ui/streamlit_app.py      # API at BENEFITPILOT_API_URL (default :8000)

Chat on the left, the selected member's year-to-date usage in the sidebar,
and under every answer a "Why this answer?" panel: the route and why, the raw
System 1 probabilities against the thresholds the policy used, the retrieved
plan sections, the rules-engine computation, the output check, guardrails,
latency and cost. All of it comes from the API response and the audit record.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
import charts

ROUTE_LABELS = {
    "claim_check": "Claim check · rules engine",
    "coverage": "Coverage · plan booklet",
    "procedure": "How-to · plan booklet",
    "emergency": "Emergency · safety message",
    "refuse_medical": "Medical advice · declined",
    "out_of_scope": "Out of scope",
    "escalate": "Escalated to a human",
}
ROUTE_SHORT = {  # fits an st.metric without truncation
    "claim_check": "Claim check",
    "coverage": "Coverage",
    "procedure": "How-to",
    "emergency": "Emergency",
    "refuse_medical": "Declined",
    "out_of_scope": "Out of scope",
    "escalate": "Escalated",
}
EXAMPLES = [
    "Is physiotherapy covered under my plan?",
    "I've used $420 of physio this year. Will a $120 session be covered?",
    "How do I submit a claim for dental?",
    "My chest hurts, what medication should I take?",
]
SYNTHETIC_NOTE = (
    "**Synthetic data only.** Every plan, member and claim here is fictional. "
    "BenefitPilot explains what a plan covers; it never gives medical advice."
)

st.set_page_config(page_title="BenefitPilot", page_icon="🩺", layout="wide")


@st.cache_resource
def get_api() -> api_client.BenefitPilotAPI:
    return api_client.BenefitPilotAPI()


def money(value) -> str:
    """Plain text (st.metric values and dataframes are not Markdown)."""
    return f"${float(value):,.2f}"


def md(text: str) -> str:
    """Escape `$` for Markdown: Streamlit renders text between two `$` as LaTeX,
    which turned "$80.00 ... $40.00" into math in the first live check."""
    return text.replace("$", "\\$")


def money_md(value) -> str:
    return md(money(value))


# --- sidebar: member + YTD usage -------------------------------------------------


def render_sidebar(api: api_client.BenefitPilotAPI) -> str | None:
    st.sidebar.title("🩺 BenefitPilot")
    st.sidebar.caption("Coverage answers and claim checks for plan members.")
    try:
        members = api.members()
        health = api.health()
    except api_client.APIError as e:
        st.sidebar.error(md(str(e)))
        return None

    options = [None] + [m["member_id"] for m in members]
    by_id = {m["member_id"]: m for m in members}

    def label(mid):
        if mid is None:
            return "No member (general plan questions)"
        m = by_id[mid]
        return f"{mid} · {m['name']} · {m['plan_name']}"

    member_id = st.sidebar.selectbox("Member", options, format_func=label, key="member_id")
    if member_id:
        render_usage(api, member_id)
    st.sidebar.divider()
    st.sidebar.caption(
        f"Router: **{health['router_backend']}** · LLM: **{health['llm_provider']}** · "
        f"v{health['version']}"
    )
    st.sidebar.info(SYNTHETIC_NOTE)
    return member_id


def render_usage(api: api_client.BenefitPilotAPI, member_id: str) -> None:
    try:
        data = api.usage(member_id)
    except api_client.APIError as e:
        st.sidebar.error(md(str(e)))
        return
    s = data["summary"]
    st.sidebar.subheader(data["name"])
    st.sidebar.caption(
        f"{s['plan_id']} ({s['plan_name']}) · {s['relationship']} · age {s['age']} · "
        f"covered since {s['coverage_start']}"
    )
    if not s["dependent_eligible"]:
        st.sidebar.warning(
            "This dependent is over the plan's age limit, so claims will be declined."
        )
    if float(s["deductible_annual"]) > 0:
        st.sidebar.caption(
            f"Deductible: {money_md(s['deductible_remaining'])} left of "
            f"{money_md(s['deductible_annual'])} this year"
        )

    st.sidebar.markdown(f"**Used in {s['benefit_year']}**")
    for c in s["categories"]:
        if c["annual_max"] is None:
            continue
        used, cap = float(c["used"]), float(c["annual_max"])
        left = money_md(c["remaining"])
        st.sidebar.progress(
            min(used / cap, 1.0) if cap else 0.0,
            text=f"{c['label']}: {money_md(used)} of {money_md(cap)} · {left} left",
        )
    for g in s["groups"]:
        used, cap = float(g["used"]), float(g["combined_max"])
        left = money_md(g["remaining"])
        st.sidebar.progress(
            min(used / cap, 1.0),
            text=f"{g['label']} (combined): {money_md(used)} of {money_md(cap)} · {left} left",
        )
    notes = [
        f"{c['label']}: next eligible {c['next_eligible']}"
        for c in s["categories"]
        if c["next_eligible"]
    ] + [
        f"{c['label']}: waiting period until {c['waiting_until']}"
        for c in s["categories"]
        if c["waiting_until"]
    ]
    if notes:
        st.sidebar.caption("  \n".join(notes))


# --- "Why this answer?" -----------------------------------------------------------


def render_why(resp: dict, audit: dict | None, thresholds: dict) -> None:
    routing = resp["decisions"]["routing"]
    ink = charts.ink_for(st.context.theme.type)
    cols = st.columns(4)
    cols[0].metric("Route", ROUTE_SHORT.get(resp["route"], resp["route"]))
    router = (audit or {}).get("router", {})
    cols[1].metric("System 1", router.get("name", "?"), help=router.get("model") or None)
    cols[2].metric("Latency", f"{resp['latency_ms'] / 1000:.1f} s")
    cols[3].metric("Cost", f"${resp['cost_usd']:.5f}")
    st.caption(f"**Why this route:** {md(resp['route_reason'])}")
    if router.get("error"):
        st.warning(f"Jev failed, so the fallback router answered: {router['error']}")

    st.markdown("##### System 1 judgments")
    left, right = st.columns(2)
    left.altair_chart(
        charts.intent_chart(routing["intent"], thresholds["auto_route_min"], ink), width="content"
    )
    right.altair_chart(
        charts.urgency_chart(routing["urgency"], thresholds["emergency_min"], ink), width="content"
    )
    st.altair_chart(
        charts.noul_chart(
            [
                (
                    "asks_medical_advice",
                    routing["asks_medical_advice"]["p"],
                    thresholds["refuse_min"],
                    f"refuse ≥ {thresholds['refuse_min']:.0%}",
                ),
                (
                    "needs_member_data",
                    routing["needs_member_data"]["p"],
                    0.5,
                    "look up member ≥ 50%",
                ),
            ],
            "Yes/no signals (Noul)",
            ink,
        ),
        width="content",
    )
    if audit and audit.get("claim_args"):
        a = audit["claim_args"]
        st.caption(
            f"**Claim arguments selected by System 1:** service `{a['category']}` "
            f"(p={a['category_p']:.2f}), amount `{a['amount']}` (p={a['amount_p']:.2f})"
        )

    if resp.get("rule_result"):
        render_rule_result(resp["rule_result"])
    if resp.get("retrieved"):
        render_retrieval(resp, audit)

    out = resp["decisions"].get("output_check")
    if out:
        st.markdown("##### Output check (System 1, after the LLM)")
        st.altair_chart(
            charts.noul_chart(
                [
                    (
                        "answer_contains_medical_advice",
                        out["answer_contains_medical_advice"]["p"],
                        thresholds["output_medical_advice_max"],
                        f"flag ≥ {thresholds['output_medical_advice_max']:.0%}",
                    ),
                    (
                        "answer_supported_by_context",
                        out["answer_supported_by_context"]["p"],
                        thresholds["output_grounded_min"],
                        f"must be ≥ {thresholds['output_grounded_min']:.0%}",
                    ),
                ],
                "Answer checks",
                ink,
            ),
            width="content",
        )
    if resp["guardrails"]:
        st.markdown("##### Guardrails that fired")
        for g in resp["guardrails"]:
            st.markdown(f"- {md(g)}")

    if audit:
        with st.container():
            st.markdown("##### Trace")
            llm = audit.get("llm")
            if llm:
                st.caption(
                    f"LLM: **{llm['provider']} / {llm['model']}** · {llm['input_tokens']} in / "
                    f"{llm['output_tokens']} out tokens · {llm['latency_ms'] / 1000:.1f} s"
                )
            stages = {k: v for k, v in audit["latency_ms"].items() if k != "total"}
            st.caption(" · ".join(f"{k}: {v:,.0f} ms" for k, v in stages.items()))
            st.caption(
                f"Audit ID `{resp['audit_id']}` (full record at `GET /audit/{resp['audit_id']}`)"
            )


def render_rule_result(r: dict) -> None:
    st.markdown("##### Rules engine (deterministic, no model)")
    cols = st.columns(5)
    cols[0].metric("Billed", money(r["billed_amount"]))
    cols[1].metric("Deductible", money(r["deductible_applied"]))
    share = r["plan_share_before_limits"]
    cols[2].metric("Plan share before limits", money(share) if share is not None else "-")
    cols[3].metric("Plan pays", money(r["covered_amount"]))
    cols[4].metric("You pay", money(r["member_pays"]))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rule": c["rule"],
                    "Result": "✓ pass"
                    if c["passed"]
                    else ("✗ deny" if c["effect"] == "deny" else "↓ reduced"),
                    "Detail": c["detail"],
                    "Section": c["citation"] or "",
                }
                for c in r["checks"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_retrieval(resp: dict, audit: dict | None) -> None:
    retrieval = (audit or {}).get("retrieval") or {}
    st.markdown("##### Retrieved plan sections")
    gate = (
        f"top cosine {retrieval['top_cosine']:.3f}"
        if retrieval.get("top_cosine") is not None
        else f"top BM25 {retrieval.get('top_bm25', 0):.2f}"
    )
    st.caption(f"Mode: **{retrieval.get('mode', '?')}** · relevance gate on {gate}")
    if retrieval.get("degraded"):
        st.warning(f"Vector search unavailable, BM25 only: {retrieval['degraded']}")
    for h in resp["retrieved"]:
        ranks = [f"RRF {h['score']:.4f}"]
        if h["bm25_rank"]:
            ranks.append(f"BM25 #{h['bm25_rank']} ({h['bm25_score']:.2f})")
        if h["vector_rank"]:
            ranks.append(f"vector #{h['vector_rank']} (cos {h['cosine']:.3f})")
        cited = "cited" if h["citation"] in resp["citations"] else "not cited"
        with st.container(border=True):
            st.markdown(f"**{h['citation']} {h['title']}** · {cited}")
            st.caption(" · ".join(ranks))
            st.markdown(md(h["markdown"]))


# --- chat ---------------------------------------------------------------------------


def ask(api: api_client.BenefitPilotAPI, question: str, member_id: str | None) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking… (the local model runs on a CPU, so allow ~30 s)"):
        try:
            resp = api.ask(question, member_id)
            audit = api.audit(resp["audit_id"])
        except api_client.APIError as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {e}", "resp": None}
            )
            return
    st.session_state.messages.append(
        {"role": "assistant", "content": resp["answer"], "resp": resp, "audit": audit}
    )


def render_chat(api: api_client.BenefitPilotAPI, member_id: str | None, thresholds: dict) -> None:
    if not st.session_state.messages:
        st.markdown("Try one of these:")
        cols = st.columns(len(EXAMPLES))
        for col, q in zip(cols, EXAMPLES, strict=True):
            if col.button(md(q), width="stretch"):
                ask(api, q, member_id)
                st.rerun()

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(md(m["content"]))
            resp = m.get("resp")
            if resp:
                cites = " ".join(f"`{c}`" for c in resp["citations"])
                st.caption(
                    f"{ROUTE_LABELS.get(resp['route'], resp['route'])}"
                    + (f" · {cites}" if cites else "")
                )
                with st.expander(
                    "Why this answer?", expanded=(i == len(st.session_state.messages) - 1)
                ):
                    render_why(resp, m.get("audit"), thresholds)

    if question := st.chat_input("Ask about your coverage or a claim"):
        ask(api, question, member_id)
        st.rerun()


def render_escalations(api: api_client.BenefitPilotAPI) -> None:
    st.caption(
        "Questions the router wasn't confident about, or answers that failed the output check."
    )
    if st.button("Refresh"):
        st.rerun()
    try:
        rows = api.escalations()
    except api_client.APIError as e:
        st.error(md(str(e)))
        return
    if not rows:
        st.info("The queue is empty.")
        return
    st.dataframe(
        pd.DataFrame(rows)[["created_at", "member_id", "question", "reason", "status", "audit_id"]],
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    api = get_api()
    st.session_state.setdefault("messages", [])
    member_id = render_sidebar(api)
    try:
        thresholds = api.thresholds()
    except api_client.APIError as e:
        st.error(md(str(e)))
        return
    chat_tab, queue_tab = st.tabs(["Chat", "Escalation queue"])
    with chat_tab:
        render_chat(api, member_id, thresholds)
    with queue_tab:
        render_escalations(api)
    st.caption(SYNTHETIC_NOTE)


main()
