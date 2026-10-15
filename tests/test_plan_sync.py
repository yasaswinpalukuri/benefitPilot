"""The plan booklets (Markdown) and the plan rules (YAML) must agree exactly.

Every number in a booklet lives in a two-column table under a `§x.y`
heading. This test parses those tables and compares them with tables
rendered from the YAML. A missing, extra or different row fails the test.
Prose deliberately avoids repeating numbers, so the tables are the
only place a number can drift (docs/DECISIONS.md, ADR-002).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benefitpilot.config import PLAN_DOCS_DIR, PLAN_RULES_DIR
from benefitpilot.rules.plan_schema import Category, Plan, load_plan

SECTION_RE = re.compile(r"^#{1,6}\s+§([\d.]+)\s+(.+?)\s*$")
PAIRS = sorted(PLAN_RULES_DIR.glob("*.yaml"))


def money(v: float) -> str:
    return f"${v:,.0f}"


def category_rows(c: Category) -> dict[str, str]:
    rows = {
        "Plan pays": f"{round(c.coinsurance * 100)}%",
        "Deductible applies": "Yes" if c.deductible_applies else "No",
    }
    if c.annual_max is not None:
        rows["Annual maximum"] = money(c.annual_max)
    if c.per_visit_max is not None:
        rows["Per-visit maximum"] = money(c.per_visit_max)
    if c.frequency is not None:
        rows["Frequency limit"] = f"{c.frequency.count} per {c.frequency.months} months"
    if c.age_limit is not None:
        rows["Age limit"] = f"Under age {c.age_limit}"
    if c.lifetime_max is not None:
        rows["Lifetime maximum"] = money(c.lifetime_max)
    if c.per_trip_max is not None:
        rows["Per-trip maximum"] = money(c.per_trip_max)
    if c.max_trip_days is not None:
        rows["Maximum trip length"] = f"{c.max_trip_days} days"
    return rows


def expected_tables(plan: Plan) -> dict[str, dict[str, str]]:
    e, d = plan.eligibility, plan.deductible
    tables = {
        e.section: {
            "Child age limit": f"Under age {e.child_age_limit}",
            "Full-time student age limit": f"Under age {e.student_age_limit}",
        },
        d.section: {
            "Annual deductible per person": (
                money(d.annual_per_person) if d.annual_per_person else "None"
            )
        },
        plan.waiting_periods.section: {
            plan.categories[k].label: f"{m} months" for k, m in plan.waiting_periods.months.items()
        },
        plan.preauthorization.section: {
            plan.categories[k].label: "All claims" if t == 0 else f"Claims over {money(t)}"
            for k, t in plan.preauthorization.thresholds.items()
        },
        plan.exclusions.section: {x.code: x.label for x in plan.exclusions.items},
    }
    for g in plan.groups.values():
        tables[g.section] = {
            "Combined annual maximum": (
                money(g.combined_annual_max)
                if g.combined_annual_max is not None
                else "No combined maximum"
            )
        }
    for c in plan.categories.values():
        tables[c.section] = category_rows(c)
    return tables


def parse_markdown(text: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Return ({section: heading title}, {section: {col0: col1}}) from a booklet."""
    headings: dict[str, str] = {}
    tables: dict[str, dict[str, str]] = {}
    section: str | None = None
    in_table = False
    for line in text.splitlines():
        if m := SECTION_RE.match(line):
            section = m.group(1)
            assert section not in headings, f"duplicate section §{section}"
            headings[section] = m.group(2)
            in_table = False
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if not in_table:  # header row starts a new table
            assert section is not None, "table before the first § heading"
            assert section not in tables, f"§{section} has more than one table"
            tables[section] = {}
            in_table = True
            continue
        if all(set(c) <= set("-: ") for c in cells):  # separator row
            continue
        assert len(cells) == 2, f"§{section}: expected 2 columns, got {cells}"
        assert cells[0] not in tables[section], f"§{section}: duplicate row {cells[0]!r}"
        tables[section][cells[0]] = cells[1]
    return headings, tables


@pytest.fixture(params=PAIRS, ids=[p.stem for p in PAIRS])
def plan_and_doc(request) -> tuple[Plan, str]:
    yaml_path: Path = request.param
    md_path = PLAN_DOCS_DIR / f"{yaml_path.stem}.md"
    assert md_path.exists(), f"no booklet for {yaml_path.name}"
    return load_plan(yaml_path), md_path.read_text(encoding="utf-8")


def test_every_rule_table_matches_yaml(plan_and_doc):
    plan, text = plan_and_doc
    _, tables = parse_markdown(text)
    assert tables == expected_tables(plan)


def test_section_titles_match_yaml_labels(plan_and_doc):
    plan, text = plan_and_doc
    headings, _ = parse_markdown(text)
    for item in [*plan.categories.values(), *plan.groups.values()]:
        assert headings.get(item.section) == item.label, f"§{item.section}"


def test_booklet_title_names_plan(plan_and_doc):
    plan, text = plan_and_doc
    assert text.splitlines()[0].startswith(f"# {plan.plan_id} — {plan.name}")


def test_booklet_declares_synthetic(plan_and_doc):
    _, text = plan_and_doc
    assert "Synthetic document" in text


def test_every_plan_has_a_booklet_and_vice_versa():
    assert {p.stem for p in PAIRS} == {p.stem for p in PLAN_DOCS_DIR.glob("*.md")}


def test_parser_catches_drift():
    """Guard against a parser that silently accepts anything."""
    plan = load_plan(PAIRS[0])
    text = (PLAN_DOCS_DIR / f"{PAIRS[0].stem}.md").read_text(encoding="utf-8")
    drifted = text.replace("| Annual maximum | $500 |", "| Annual maximum | $600 |", 1)
    assert drifted != text
    assert parse_markdown(drifted)[1] != expected_tables(plan)
