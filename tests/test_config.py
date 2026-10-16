from __future__ import annotations

import pytest

from benefitpilot.config import Settings


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_api_keys_count_as_missing(blank, monkeypatch):
    """`.env.example` ships `TYPESAFE_API_KEY=`. A blank key once slipped past an
    `is None` check and reached the SDK, which crashed a benchmark run."""
    monkeypatch.setenv("TYPESAFE_API_KEY", blank)
    monkeypatch.setenv("ANTHROPIC_API_KEY", blank)
    s = Settings(_env_file=None)
    assert s.typesafe_api_key is None
    assert s.anthropic_api_key is None


def test_real_key_is_kept_secret(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-example-123")
    s = Settings(_env_file=None)
    assert s.typesafe_api_key.get_secret_value() == "ts-example-123"
    assert "ts-example-123" not in repr(s)
