from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from benefitpilot.config import PLAN_RULES_DIR
from benefitpilot.rules.plan_schema import Plan, load_plans


def test_both_plans_load():
    plans = load_plans()
    assert set(plans) == {"PLAN-A", "PLAN-B"}
    assert plans["PLAN-A"].name == "Bronze"
    assert plans["PLAN-B"].name == "Gold"


def test_citation_format():
    plan = load_plans()["PLAN-A"]
    assert plan.citation(plan.categories["physiotherapy"].section) == "PLAN-A §4.2.1"


def test_gold_is_at_least_as_generous_as_bronze():
    a, b = load_plans()["PLAN-A"], load_plans()["PLAN-B"]
    for name, cat_a in a.categories.items():
        cat_b = b.categories[name]
        assert cat_b.coinsurance >= cat_a.coinsurance, name
        if cat_a.annual_max is not None and cat_b.annual_max is not None:
            assert cat_b.annual_max >= cat_a.annual_max, name


def _raw(stem: str) -> dict:
    return yaml.safe_load((PLAN_RULES_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))


def test_rejects_unknown_field():
    raw = _raw("plan_a")
    raw["categories"]["physiotherapy"]["anual_max"] = 1  # typo
    with pytest.raises(ValidationError):
        Plan.model_validate(raw)


def test_rejects_waiting_period_for_unknown_category():
    raw = _raw("plan_a")
    raw["waiting_periods"]["months"]["dentl_major"] = 12
    with pytest.raises(ValidationError, match="unknown categories"):
        Plan.model_validate(raw)


def test_rejects_category_that_is_also_excluded():
    raw = _raw("plan_b")
    raw["exclusions"]["items"].append({"code": "orthodontics", "label": "x"})
    with pytest.raises(ValidationError, match="excluded"):
        Plan.model_validate(raw)


def test_rejects_coinsurance_over_100_percent():
    raw = _raw("plan_a")
    raw["categories"]["physiotherapy"]["coinsurance"] = 1.2
    with pytest.raises(ValidationError):
        Plan.model_validate(raw)
