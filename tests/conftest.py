from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from databases import Database
from fastapi.testclient import TestClient


def _count_rows_sqlite(db_path: Path) -> int:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("select count(*) from predictions")
        return int(cur.fetchone()[0])
    finally:
        con.close()


class DummyModel:
    def predict(self, X):
        return [1.23] * len(X)


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    from src.api import main as api_main

    db_path = tmp_path / "predictions_test.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    api_main.database = Database(db_url)

    api_main.model = DummyModel()

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

    with TestClient(api_main.app) as client:
        yield client, db_path, api_main


@pytest.fixture()
def count_predictions():
    return _count_rows_sqlite
