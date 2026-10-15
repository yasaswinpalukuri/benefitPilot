from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from benefitpilot.api.app import app
from benefitpilot.api.deps import get_store
from benefitpilot.store import MemberStore
from scripts.generate_data import generate, write_sqlite


@pytest.fixture(scope="session")
def dataset():
    return generate()


@pytest.fixture(scope="session")
def db_path(tmp_path_factory, dataset):
    path = tmp_path_factory.mktemp("db") / "benefitpilot.db"
    write_sqlite(dataset, path)
    return path


@pytest.fixture(scope="session")
def store(db_path) -> MemberStore:
    return MemberStore(db_path)


@pytest.fixture
def client(store):
    app.dependency_overrides[get_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def canonical_claim(dataset):
    return next(c for c in dataset.claims if c.notes.startswith("canonical"))
