import sqlite3
from pathlib import Path

import pytest
from databases import Database
from fastapi.testclient import TestClient

from src.api import main as api_main


# symuluje model
class DummyModel:
    # zwraca predykcje
    def predict(self, X):
        return [0.0] * len(X)


# zlicza rekordy w tabeli predictions
def _count_predictions(db_path: Path) -> int:
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT COUNT(*) FROM predictions")
        return int(cur.fetchone()[0])
    finally:
        con.close()


# przygotowuje klienta testowego i bazę sqlite
@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions_test.db"

    api_main.database = Database(f"sqlite+aiosqlite:///{db_path}")
    api_main.model = DummyModel()

    monkeypatch.setattr(
        api_main.settings,
        "REQUIRED_COLUMNS_PATH",
        str(tmp_path / "missing_required_columns.json"),
        raising=False,
    )

    monkeypatch.setattr(
        api_main.settings,
        "MODEL_PATH",
        str(tmp_path / "missing_model.pkl"),
        raising=False,
    )

    with TestClient(api_main.app) as c:
        api_main.required_columns[:] = []
        yield c, db_path


# sprawdza walidację braku pola data
def test_validation_error_missing_data(client):
    c, _ = client
    r = c.post("/predict", json={"feature1": 1.23, "feature2": 4.56})
    assert r.status_code == 422


# sprawdza predykcję i zapis rekordu do bazy
def test_predict_returns_200_and_saves_record_in_db(client):
    c, db_path = client

    before = _count_predictions(db_path)

    r = c.post("/predict", json={"data": {"feature1": 1.23, "feature2": 4.56}})
    assert r.status_code == 200

    body = r.json()
    assert "prediction" in body
    assert "model_version" in body

    after = _count_predictions(db_path)
    assert after == before + 1


# sprawdza endpoint healthz
def test_health_check(client):
    c, _ = client

    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
