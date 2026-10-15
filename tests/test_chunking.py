from __future__ import annotations

import pytest

from benefitpilot.retrieval.chunking import chunk_markdown, load_chunks
from benefitpilot.rules.plan_schema import load_plans

CHUNKS = load_chunks()
BY_ID = {c.id: c for c in CHUNKS}


def test_chunk_ids_are_citation_ids_and_unique():
    assert len(BY_ID) == len(CHUNKS)
    assert all(c.id == f"{c.plan_id} §{c.section}" for c in CHUNKS)


def test_every_rule_section_is_retrievable():
    """Each section the rules engine can cite must exist as a chunk."""
    for plan in load_plans().values():
        sections = [c.section for c in plan.categories.values()]
        sections += [g.section for g in plan.groups.values()]
        sections += [
            plan.eligibility.section,
            plan.deductible.section,
            plan.waiting_periods.section,
            plan.preauthorization.section,
            plan.exclusions.section,
        ]
        for s in sections:
            assert plan.citation(s) in BY_ID


def test_heading_only_sections_become_parent_context():
    assert "PLAN-A §4" not in BY_ID  # "Health benefits" has no body of its own
    physio = BY_ID["PLAN-A §4.2.1"]
    assert physio.parents == ("§4 Health benefits", "§4.2 Paramedical services")
    assert "Part of: §4 Health benefits > §4.2 Paramedical services" in physio.text
    assert physio.text.startswith("PLAN-A (Bronze plan) — §4.2.1 Physiotherapy")


def test_tables_are_flattened_for_indexing_and_kept_for_display():
    physio = BY_ID["PLAN-A §4.2.1"]
    assert "Annual maximum: $500" in physio.text
    assert "| Annual maximum | $500 |" in physio.markdown
    assert "Major dental — Waiting period: 12 months" in BY_ID["PLAN-A §7"].text
    assert "orthodontics — Excluded service: Orthodontic treatment" in BY_ID["PLAN-A §8"].text


def test_synthetic_notice_is_not_indexed():
    assert not any("Synthetic document" in c.text for c in CHUNKS)


def test_rejects_booklet_without_plan_title():
    with pytest.raises(ValueError, match="PLAN-X"):
        chunk_markdown("# Some booklet\n\n## §1 Intro\n\ntext")
