"""Thin HTTP client for the BenefitPilot API.

The UI is a pure API client: everything the "Why this answer?" panel shows is
what the API returned and the audit log recorded, never something the UI
recomputed.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:8000"
# The local model runs on a CPU; a claim or coverage answer can take ~30 s.
ASK_TIMEOUT_S = float(os.getenv("BENEFITPILOT_ASK_TIMEOUT_S", "600"))


class APIError(RuntimeError):
    pass


class BenefitPilotAPI:
    def __init__(self, base_url: str | None = None, transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or os.getenv("BENEFITPILOT_API_URL", DEFAULT_URL)).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=15.0, transport=transport)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            r = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise APIError(f"Cannot reach the BenefitPilot API at {self.base_url}: {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise APIError(f"{r.status_code}: {detail}")
        return r.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def thresholds(self) -> dict:
        return self._request("GET", "/thresholds")

    def members(self) -> list[dict]:
        return self._request("GET", "/members")

    def usage(self, member_id: str) -> dict:
        return self._request("GET", f"/members/{member_id}/usage")

    def ask(self, question: str, member_id: str | None) -> dict:
        return self._request(
            "POST",
            "/ask",
            json={"question": question, "member_id": member_id},
            timeout=ASK_TIMEOUT_S,
        )

    def audit(self, audit_id: str) -> dict:
        return self._request("GET", f"/audit/{audit_id}")

    def escalations(self) -> list[dict]:
        return self._request("GET", "/escalations")
