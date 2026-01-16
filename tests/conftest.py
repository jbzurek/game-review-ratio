from __future__ import annotations
import pytest
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.testclient import TestClient
from databases import Database
import sqlite3

from src.api import main as api_main


# ----------------------------
# Dummy model for testing
# ----------------------------
class DummyModel:
    def predict(self, x):
        import pandas as pd

        return pd.Series([42.0])  # always returns 42.0


# ----------------------------
# SQLite helper
# ----------------------------
def _count_rows_sqlite(db_path: Path) -> int:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT count(*) FROM predictions")
        return int(cur.fetchone()[0])
    finally:
        con.close()


# ----------------------------
# API Client fixture
# ----------------------------
@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    # dummy lifespan as async context manager
    @asynccontextmanager
    async def dummy_lifespan(app):
        # Hardcode model path & dummy model
        api_main.model_path = Path("data/06_models/production_model.pkl")
        api_main.model = DummyModel()
        api_main.model_type = "sklearn"
        api_main.required_columns[:] = [
            "required_age",
            "price",
            "dlc_count",
            "windows",
            "mac",
            "linux",
            "metacritic_score",
            "achievements",
            "discount",
            "release_year",
            "release_month",
        ]

        # Setup fake DB
        db_path = tmp_path / "predictions_test.db"
        api_main.database = Database(f"sqlite+aiosqlite:///{db_path}")
        await api_main.database.connect()
        await api_main.init_db()
        try:
            yield
        finally:
            await api_main.database.disconnect()

    # patch the lifespan
    monkeypatch.setattr(api_main, "lifespan", dummy_lifespan)
    api_main.app.router.lifespan_context = lambda app: dummy_lifespan(app)

    # return TestClient
    with TestClient(api_main.app) as client:
        yield client, tmp_path / "predictions_test.db", api_main


# ----------------------------
# Count predictions fixture
# ----------------------------
@pytest.fixture()
def count_predictions():
    return _count_rows_sqlite
