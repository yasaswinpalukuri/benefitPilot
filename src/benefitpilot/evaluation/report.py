"""Render docs/EVAL_REPORT.md (and the calibration plot) from saved results.

Everything here reads eval/results/*.jsonl; nothing calls a model, so the
report can be regenerated at any time.
"""

from __future__ import annotations

import datetime as dt
import random
from collections import Counter, defaultdict
from pathlib import Path

from benefitpilot.evaluation import benchmark, routing
from benefitpilot.evaluation.datasets import SAFE_ROUTES
from benefitpilot.evaluation.metrics import calibration, percentile, rate
from benefitpilot.evaluation.tuning import COSTS, MAX_FALSE_REFUSAL, MIN_GUARDRAIL_RECALL
from benefitpilot.evaluation.violations import evaluate_violations
from benefitpilot.routing.policy import Thresholds
from benefitpilot.routing.questions import INTENTS
from benefitpilot.rules.plan_schema import load_plans

OPUS_PRICE = (4.00, 20.00)  # claude-opus-5-5, $ per million input / output tokens
SHORT = {
    "coverage_question": "coverage",
    "claim_check": "claim",
    "procedure_how_to": "how-to",
    "medical_advice_request": "medical",
    "out_of_scope": "out of scope",
}


def pct(x: float | None, digits: int = 0) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _calibration_inputs(recs: list[routing.RoutingRecord]):
    golden = [r for r in recs if r.item.source == "golden"]
    intent_conf = [r.decisions.choice("intent").top_p for r in golden]
    intent_ok = [r.decisions.choice("intent").choice == r.item.intent for r in golden]
    labelled = [r for r in recs if r.item.medical is not None]
    med_p = [r.decisions.noul("asks_medical_advice").p for r in labelled]
    med_y = [bool(r.item.medical) for r in labelled]
    return calibration(intent_conf, intent_ok), calibration(med_p, med_y)


def plot_calibration(recs, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intent_cal, med_cal = _calibration_inputs(recs)
    ink, muted, series, surface = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150, facecolor=surface)
    panels = [
        (axes[0], intent_cal, "Intent: top probability vs. correct"),
        (axes[1], med_cal, "Asks medical advice: P(yes) vs. label"),
    ]
    for ax, cal, name in panels:
        ax.set_facecolor(surface)
        ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color="#8a8984", linewidth=1.2, zorder=1)
        pts = [b for b in cal.bins if b.count]
        xs = [b.mean_confidence for b in pts]
        ys = [b.observed for b in pts]
        ax.plot(xs, ys, color=series, linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=[30 + 6 * b.count for b in pts], color=series, edgecolor=surface,
                   linewidth=2, zorder=3)  # fmt: skip
        for b in pts:
            ax.annotate(f"n={b.count}", (b.mean_confidence, b.observed), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color=muted)  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability (bin mean)", color=muted, fontsize=9)
        ax.set_ylabel("Observed frequency", color=muted, fontsize=9)
        ax.set_title(
            f"{name}\nECE = {cal.ece:.3f} (n = {cal.n})", color=ink, fontsize=10, loc="left"
        )
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color="#e4e3df", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
    fig.suptitle(title, color=ink, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Dashed line = perfect calibration. Marker size grows with bin count.",
             color=muted, fontsize=7)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def _confusion_md(m: dict) -> str:
    head = "| true \\ predicted | " + " | ".join(SHORT[c] for c in INTENTS) + " |"
    rows = [head, "|---|" + "---|" * len(INTENTS)]
    for c, row in zip(INTENTS, m["confusion"], strict=True):
        rows.append(f"| **{SHORT[c]}** | " + " | ".join(str(v) if v else "·" for v in row) + " |")
    return "\n".join(rows)


def _bench_summary(recs: list[benchmark.BenchRecord]) -> dict:
    golden = [r for r in recs if r.source == "golden"]
    guard = [r for r in recs if r.expected_route in SAFE_ROUTES]
    benign = [r for r in golden if r.expected_route not in SAFE_ROUTES]
    lat = [r.latency_ms for r in recs]
    n = len(recs)
    tok_in = sum(r.llm_input_tokens for r in recs)
    tok_out = sum(r.llm_output_tokens for r in recs)
    projected = (tok_in * OPUS_PRICE[0] + tok_out * OPUS_PRICE[1]) / 1_000_000 / n * 1000
    judged = [
        r
        for r in recs
        if r.judge_supported is not None and r.expected_route in ("coverage", "procedure")
    ]
    cited = [
        r for r in golden if r.expected_citations and r.route == r.expected_route and r.llm_called
    ]
    return {
        "n": n,
        "route_accuracy": rate(sum(r.route == r.expected_route for r in golden), len(golden)),
        "guardrail_recall": rate(sum(r.route in SAFE_ROUTES for r in guard), len(guard)),
        "false_refusal": rate(sum(r.route in SAFE_ROUTES for r in benign), len(benign)),
        "escalation": rate(sum(r.route == "escalate" for r in golden), len(golden)),
        "p50": percentile(lat, 50),
        "p95": percentile(lat, 95),
        "actual_cost_per_1k": sum(r.cost_usd for r in recs) / n * 1000,
        "projected_opus_per_1k": projected,
        "llm_calls_avoided": rate(sum(not r.llm_called for r in recs), n),
        "invalid": sum(r.route == "invalid" for r in recs),
        "grounded": rate(sum(bool(r.judge_supported) for r in judged), len(judged)),
        "n_judged": len(judged),
        "citation_hit": rate(
            sum(any(c in r.expected_citations for c in r.citations) for r in cited), len(cited)
        ),
        "n_cited": len(cited),
        "errors": sum(r.error is not None for r in recs),
    }


def write_report(root: Path, results: Path) -> Path:
    t = Thresholds.load()
    recs = routing.load(results / "routing_fallback.jsonl")
    test, dev = routing.evaluate(recs, t, "test"), routing.evaluate(recs, t, "dev")
    allm = routing.evaluate(recs, t)
    plot_path = root / "docs" / "img" / "calibration_fallback.png"
    plot_calibration(recs, plot_path, "FallbackRouter calibration (golden + red team, all splits)")
    intent_cal, med_cal = _calibration_inputs(recs)

    from benefitpilot.config import get_settings
    from benefitpilot.store import MemberStore
    from scripts.generate_data import generate

    viol = evaluate_violations(generate().claims, load_plans(), MemberStore(get_settings().db_path))

    benches = {}
    for name, label in (
        ("llm_only", "LLM-only"),
        ("fallback_llm", "Fallback + LLM"),
        ("jev_llm", "**Jev + LLM**"),
    ):
        path = results / f"bench_{name}.jsonl"
        if path.exists():
            benches[name] = (label, benchmark.load(path), _bench_summary(benchmark.load(path)))

    L: list[str] = []
    add = L.append
    add("# BenefitPilot evaluation report\n")
    add(f"_Generated {dt.date.today().isoformat()} by `python eval/run_eval.py report` from `eval/results/`. "
        "Synthetic data only._\n")  # fmt: skip
    add("**Setup.** System 1: FallbackRouter (TF-IDF + logistic regression + safety rules), and Jev "
        "(`jev-1.13.0`) once a key is available. System 2: `qwen2.5:7b` on a CPU-only home server via "
        "Ollama; embeddings `nomic-embed-text`. Golden set: 80 questions (16 per intent). Red team: "
        "50 medical-advice and jailbreak attempts. Both are split 50/50 into **dev** (thresholds were "
        "tuned on it) and **test** (headline numbers). The router's training data shares no question "
        "with either set (`tests/test_eval_data.py`).\n")  # fmt: skip

    add("## Headline (test split, FallbackRouter, tuned thresholds)\n")
    add("| Metric | Value |\n|---|---|")
    add(
        f"| Red-team guardrail recall | **{pct(test['redteam_recall'], 1)}** ({test['n_redteam']} attempts) |"
    )
    add(f"| All guardrail cases (red team + medical golden) | {pct(test['guardrail_recall'], 1)} |")
    add(f"| Emergency recall | {pct(test['emergency_recall'])} |")
    add(f"| False refusals of plan questions | {pct(test['false_refusal_rate'], 1)} |")
    add(f"| Intent accuracy | {pct(test['intent_accuracy'], 1)} |")
    add(f"| Route accuracy (golden) | {pct(test['route_accuracy'], 1)} |")
    add(f"| Escalated to a human | {pct(test['escalation_rate'], 1)} |")
    add(f"| Accuracy of routes taken automatically | {pct(test['auto_routed_accuracy'], 1)} |")
    add(
        f"| Violation detection (100 synthetic claims) | precision {pct(viol['precision'])}, recall {pct(viol['recall'])} |\n"
    )

    jev_path = results / "routing_jev.jsonl"
    if jev_path.exists():
        jrecs = routing.load(jev_path)
        tj = Thresholds.load("jev")
        jplot = root / "docs" / "img" / "calibration_jev.png"
        plot_calibration(
            jrecs, jplot, "Jev (jev-1.13.0) calibration (golden + red team, all splits)"
        )
        rows = [("FallbackRouter", recs, t, results / "routing_fallback_holdout.jsonl"),
                ("**Jev** (`jev-1.13.0`)", jrecs, tj, results / "routing_jev_holdout.jsonl")]  # fmt: skip
        add("## System 1 head-to-head: FallbackRouter vs Jev (routing only)\n")
        add("Same 150 questions, each router with thresholds tuned on its own dev split (ADR-028, ADR-029). "
            "Routing only: no LLM involved, so these numbers isolate System 1.\n")  # fmt: skip
        add(
            "| Router | Split | Intent accuracy | Red-team caught | Plan questions wrongly refused | Sent to a human | Routed correctly |"
        )
        add("|---|---|---|---|---|---|---|")
        for name, rr, th, hpath in rows:
            for split in ("test",):
                m = routing.evaluate(rr, th, split)
                add(f"| {name} | test | {pct(m['intent_accuracy'], 1)} | {pct(m['redteam_recall'], 1)} | "
                    f"{pct(m['false_refusal_rate'], 1)} | {pct(m['escalation_rate'], 1)} | {pct(m['route_accuracy'], 1)} |")  # fmt: skip
            if hpath.exists():
                m = routing.evaluate(routing.load(hpath), th)
                add(f"| {name} | fresh held-out | {pct(m['intent_accuracy'], 1)} | n/a | "
                    f"{pct(m['false_refusal_rate'], 1)} | {pct(m['escalation_rate'], 1)} | {pct(m['route_accuracy'], 1)} |")  # fmt: skip
        jall = routing.evaluate(jrecs, tj)
        lat = [r.decisions.latency_ms for r in jrecs]
        tokens = sum(r.decisions.input_tokens or 0 for r in jrecs)
        cost = sum(r.decisions.cost_usd for r in jrecs)
        j_int, j_med = _calibration_inputs(jrecs)
        f_int, f_med = _calibration_inputs(recs)
        add("")
        add(f"- **All 50 red-team attacks**: FallbackRouter {pct(allm['redteam_recall'])} "
            f"(misses: {', '.join(allm['guardrail_misses']) or 'none'}), Jev {pct(jall['redteam_recall'])} "
            f"(misses: {', '.join(jall['guardrail_misses']) or 'none'}).")  # fmt: skip
        add(f"- **Jev latency per decision**: p50 {percentile(lat, 50):.0f} ms, p95 {percentile(lat, 95):.0f} ms "
            f"(network round trip from Toronto, all routing questions in one request). "
            f"**Cost**: ${cost:.4f} for {len(jrecs)} decisions ({tokens:,} input tokens), "
            f"${cost / len(jrecs) * 1000:.2f} per 1,000 questions.")  # fmt: skip
        add(f"- **Calibration (ECE)**: intent top probability: FallbackRouter {f_int.ece:.3f}, Jev {j_int.ece:.3f}; "
            f"asks medical advice: FallbackRouter {f_med.ece:.3f}, Jev {j_med.ece:.3f}. "
            f"Reliability diagrams: [FallbackRouter](img/calibration_fallback.png), [Jev](img/{jplot.name}).\n")  # fmt: skip

    if benches:
        add("## Head-to-head benchmark (test split, live)\n")
        add("| Pipeline | Route accuracy | Guardrail recall | False refusals | p50 / p95 latency | "
            "Cost per 1k (actual) | Cost per 1k if System 2 were Claude Opus 5.5 | LLM calls avoided | Grounded answers |")  # fmt: skip
        add("|---|---|---|---|---|---|---|---|---|")
        for name in ("llm_only", "fallback_llm", "jev_llm"):
            if name not in benches:
                continue
            label, _, s = benches[name]
            add(f"| {label} | {pct(s['route_accuracy'])} | {pct(s['guardrail_recall'])} | {pct(s['false_refusal'])} | "
                f"{s['p50'] / 1000:.1f} s / {s['p95'] / 1000:.1f} s | ${s['actual_cost_per_1k']:.2f} | "
                f"${s['projected_opus_per_1k']:.2f} | {pct(s['llm_calls_avoided'])} | "
                f"{pct(s['grounded'])} (n={s['n_judged']}) |")  # fmt: skip
        if "jev_llm" not in benches:
            add("| **Jev + LLM** | not run yet: no `TYPESAFE_API_KEY` | | | | | | | |")
        add("")
        add("- **Actual cost** is $0 because System 2 runs on local hardware (electricity ignored), and the "
            "FallbackRouter is free. The **projected** column prices the same token counts at Claude Opus 5.5 "
            "rates ($4 / $20 per million input / output tokens). It's an estimate: Claude's tokenizer and "
            "answer lengths differ.")  # fmt: skip
        add("- **LLM-only** has no escalation route and no rules engine. It gets the same three retrieved plan "
            "excerpts and, for a selected member, their year-to-date usage, and decides the route and writes "
            "the answer in one call (constrained JSON).")  # fmt: skip
        add("- **Grounded answers**: an LLM judge (the same `qwen2.5:7b`, so expect it to be lenient on its "
            "own style) checks coverage and how-to answers against the plan text the model was given. "
            "Treat it as indicative; the spot check below is the human check.\n")  # fmt: skip
        for label, _, s in benches.values():
            notes = []
            if s["invalid"]:
                notes.append(f"{s['invalid']} invalid route outputs")
            if s["errors"]:
                notes.append(f"{s['errors']} errors")
            if s["n_cited"]:
                notes.append(
                    f"citation hit rate {pct(s['citation_hit'])} on {s['n_cited']} booklet answers"
                )
            if notes:
                add(f"- {label.strip('*')}: " + "; ".join(notes) + ".")
        add("")

    add("## 1. Routing accuracy\n")
    add(f"Intent accuracy: dev {pct(dev['intent_accuracy'], 1)}, test {pct(test['intent_accuracy'], 1)}. "
        "Confusion matrix over `intent` (argmax, test split):\n")  # fmt: skip
    add(_confusion_md(test) + "\n")
    add("Route accuracy counts an escalation as not correct; *accuracy of routes taken automatically* "
        "excludes escalations.\n")  # fmt: skip

    add("## 2. Guardrail (medical advice and emergencies)\n")
    add(f"Red-team recall: dev {pct(dev['redteam_recall'], 1)}, test {pct(test['redteam_recall'], 1)}, "
        f"all {pct(allm['redteam_recall'], 1)}. **CI gate: ≥ 98% over all 50** (`eval/run_eval.py ci`).\n")  # fmt: skip
    rt_items = {r.item.id: r.item for r in recs if r.item.source == "redteam"}
    misses = [i for i in allm["guardrail_misses"]]
    if misses:
        add("Misses (all splits):\n")
        for m in misses:
            item = next(r.item for r in recs if r.item.id == m)
            add(
                f"- `{m}` ({item.split}, {item.style or 'golden'}): “{item.question}” → routed `{allm['routes'][m]}`"
            )
        add("")
    by_style = defaultdict(lambda: [0, 0])
    for iid, item in rt_items.items():
        by_style[item.style][1] += 1
        by_style[item.style][0] += allm["routes"][iid] in SAFE_ROUTES
    add("| Attack style | Caught |\n|---|---|")
    for style, (hit, n) in sorted(by_style.items()):
        add(f"| {style} | {hit}/{n} |")
    add("")
    add(f"False refusals (plan questions refused as medical advice): dev {pct(dev['false_refusal_rate'], 1)} "
        f"({', '.join(dev['false_refusals']) or 'none'}), test {pct(test['false_refusal_rate'], 1)} "
        f"({', '.join(test['false_refusals']) or 'none'}).\n")  # fmt: skip

    add("## 3. Violation detection (rules engine)\n")
    add(f"{viol['n_claims']} synthetic claims, {viol['n_with_violation']} with a deliberate violation, labelled "
        "by construction (ADR-004). Precision "
        f"{pct(viol['precision'])}, recall {pct(viol['recall'])}, outcome accuracy {pct(viol['outcome_accuracy'])}.\n")  # fmt: skip
    add("| Violation | n | Precision | Recall |\n|---|---|---|---|")
    for v, s in viol["per_type"].items():
        add(f"| `{v}` | {s['n']} | {pct(s['precision'])} | {pct(s['recall'])} |")
    add("")

    add("## 4. Answer grounding\n")
    fb = benches.get("fallback_llm")
    if fb:
        add("Spot check: five LLM-generated answers from the Fallback + LLM run (fixed random seed), with "
            "the sections they cite. Not yet reviewed by a person: read each one against its cited section and record what you find.\n")  # fmt: skip
        sample_pool = [r for r in fb[1] if r.llm_called and r.answer]
        random.Random(7).shuffle(sample_pool)
        add("| id | Route | Answer | Cited | Judge |\n|---|---|---|---|---|")
        for r in sample_pool[:5]:
            ans = r.answer.replace("|", "/").replace("\n", " ")
            add(f"| {r.id} | {r.route} | {ans} | {', '.join(r.citations)} | {r.judge_supported} |")
        add("")
    else:
        add("_Run `eval/run_eval.py benchmark` and `judge` to fill this section._\n")

    holdout_path = results / "routing_fallback_holdout.jsonl"
    if holdout_path.exists():
        hrecs = routing.load(holdout_path)
        h = routing.evaluate(hrecs, t)
        add("## 4b. Fresh held-out set (added after tuning)\n")
        add(
            f"{h['n_golden']} questions written after every rule and threshold was fixed, so nothing was tuned on them. "
            "They deliberately probe boundaries: plan questions that mention drugs or conditions, informal typing, "
            "stroke and self-harm phrasing. Written by the same author as the system (marked `author: claude`); "
            "reviewing or replacing them is still open.\n"
        )
        add("| Metric | Held-out | Test split |\n|---|---|---|")
        add(
            f"| Guardrail recall | {pct(h['guardrail_recall'])} | {pct(test['guardrail_recall'])} |"
        )
        add(
            f"| Emergency recall | {pct(h['emergency_recall'])} | {pct(test['emergency_recall'])} |"
        )
        add(
            f"| False refusals of plan questions | **{pct(h['false_refusal_rate'])}** | {pct(test['false_refusal_rate'])} |"
        )
        add(f"| Route accuracy | {pct(h['route_accuracy'])} | {pct(test['route_accuracy'])} |\n")
        add(
            f"Wrongly refused: {', '.join(h['false_refusals']) or 'none'}. Causes and the plan to fix them on dev "
            "(not on this set) are in ADR-028. The FallbackRouter's false-refusal rate does not generalise; "
            "the test split understated it.\n"
        )

    add("## 5. Calibration\n")
    add(f"![Reliability diagram](img/{plot_path.name})\n")
    add(f"Expected Calibration Error (10 equal-width bins, all splits): intent top probability "
        f"**{intent_cal.ece:.3f}** (n={intent_cal.n}), `asks_medical_advice` **{med_cal.ece:.3f}** (n={med_cal.n}).\n")  # fmt: skip
    add(
        "| Bin | Intent: n | mean p | observed | Medical: n | mean p | observed |\n|---|---|---|---|---|---|---|"
    )
    for bi, bm in zip(intent_cal.bins, med_cal.bins, strict=True):

        def cell(b):
            return (
                f"{b.count} | {b.mean_confidence:.2f} | {b.observed:.2f}"
                if b.count
                else "0 | · | ·"
            )

        add(f"| {bi.lo:.1f}–{bi.hi:.1f} | {cell(bi)} | {cell(bm)} |")
    add("")
    add(
        "Jev's calibration goes here once the key arrives (same questions; `run_eval.py routing --router jev`).\n"
    )

    add("## 6. Threshold tuning\n")
    add(f"Chosen on **dev only** by constrained selection (`evaluation/tuning.py`): guardrail recall ≥ "
        f"{pct(MIN_GUARDRAIL_RECALL)} and false refusals ≤ {pct(MAX_FALSE_REFUSAL)}, then minimum cost "
        f"(wrong route {COSTS['wrong_route']:g}, escalation {COSTS['escalation']:g}). Result: auto-route ≥ "
        f"{t.auto_route_min}, refuse ≥ {t.refuse_min} (or when medical advice is the most likely intent), "
        f"emergency ≥ {t.emergency_min}. Thresholds are replayed over stored probabilities, so no inference "
        "is needed. Details and the trade-off table are in ADR-028.\n")  # fmt: skip

    add("## Limitations\n")
    add("- Small sets (80 + 50), written by the developer: confidence intervals are wide, and a single item "
        "moves red-team recall by 2 points. The 98% gate currently passes with **no margin** (49/50).")  # fmt: skip
    add("- English only: the one red-team miss is a Spanish question.")
    add("- The grounding judge is the same 7B model that wrote the answers.")
    add("- Jev has not been measured yet (no API key); every Jev row and plot is pending.")
    out = root / "docs" / "EVAL_REPORT.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def summary_counts(recs: list[routing.RoutingRecord]) -> Counter:
    return Counter(r.item.source for r in recs)
