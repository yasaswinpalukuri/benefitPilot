from fastapi.testclient import TestClient

from benefitpilot.api.app import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
