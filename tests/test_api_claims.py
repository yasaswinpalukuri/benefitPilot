from __future__ import annotations

from benefitpilot.api.deps import get_store
from benefitpilot.config import get_settings


def _body(claim, **overrides):
    body = {
        "member_id": claim.member_id,
        "category": claim.category,
        "amount": str(claim.amount),
        "date": claim.service_date.isoformat(),
    }
    return body | overrides


def test_canonical_claim(client, canonical_claim):
    r = client.post("/claims/check", json=_body(canonical_claim))
    assert r.status_code == 200
    data = r.json()
    assert data["outcome"] == "partial"
    assert data["covered_amount"] == "80.00"  # Decimal as an exact string
    assert data["violations"] == ["annual_max_exceeded"]
    assert data["citations"] == ["PLAN-A §4.2.1"]
    assert any(c["rule"] == "annual_max" and not c["passed"] for c in data["checks"])


def test_unknown_member_is_404(client, canonical_claim):
    r = client.post("/claims/check", json=_body(canonical_claim, member_id="M999"))
    assert r.status_code == 404


def test_unknown_category_is_422_with_valid_options(client, canonical_claim):
    r = client.post("/claims/check", json=_body(canonical_claim, category="physio"))
    assert r.status_code == 422
    assert "physiotherapy" in r.json()["detail"]["valid_categories"]


def test_excluded_category_is_a_normal_denial(client, canonical_claim):
    r = client.post("/claims/check", json=_body(canonical_claim, category="orthodontics"))
    assert r.status_code == 200
    assert r.json()["violations"] == ["excluded_service"]


def test_rejects_bad_amounts(client, canonical_claim):
    for amount in ("-5", "0", "12.345"):
        r = client.post("/claims/check", json=_body(canonical_claim, amount=amount))
        assert r.status_code == 422, amount


def test_missing_database_is_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "missing.db"))
    get_settings.cache_clear()
    try:
        get_store()
    except Exception as e:  # HTTPException
        assert getattr(e, "status_code", None) == 503
    else:
        raise AssertionError("expected 503")
    finally:
        get_settings.cache_clear()
