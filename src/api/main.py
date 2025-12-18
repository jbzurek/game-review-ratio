import datetime as dt
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd
from databases import Database
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ustawia ścieżkę do modelu
    MODEL_PATH: str = "data/06_models/production_model.pkl"
    # ustawia ścieżkę do wymaganych kolumn
    REQUIRED_COLUMNS_PATH: str = "data/06_models/required_columns.json"

    # ustawia url do bazy
    DATABASE_URL: str = "sqlite+aiosqlite:///./predictions.db"

    # ustawia wersję modelu
    MODEL_VERSION: str | None = None

    # ustawia klucz do wandb
    WANDB_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# tworzy połączenie do bazy
database = Database(settings.DATABASE_URL)

# trzyma model w pamięci
model = None

# trzyma listę wymaganych kolumn
required_columns: list[str] = []


def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    # liczy sha256 pliku
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_model_version() -> str:
    # bierze wersję z env albo liczy z pliku
    if settings.MODEL_VERSION:
        return settings.MODEL_VERSION
    if os.path.isfile(settings.MODEL_PATH):
        digest = _file_sha256(settings.MODEL_PATH)[:12]
        name = Path(settings.MODEL_PATH).name
        return f"{name}:{digest}"
    return "unknown"


# ustawia wersję modelu
MODEL_VERSION = _resolve_model_version()


class Features(BaseModel):
    # przyjmuje cechy jako słownik
    data: Dict[str, Union[int, float]]

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        # sprawdza pusty payload
        if not v:
            raise ValueError("payload data nie może być pusty")

        # sprawdza obce kolumny (literówki)
        if required_columns:
            unknown = set(v.keys()) - set(required_columns)
            if unknown:
                raise ValueError(f"nieznane kolumny w payload: {sorted(unknown)}")

        return v


class Prediction(BaseModel):
    # zwraca predykcję i wersję modelu
    prediction: float | str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, required_columns

    # ładuje model raz
    if model is None:
        if not os.path.isfile(settings.MODEL_PATH):
            raise RuntimeError(f"nie znaleziono pliku modelu w '{settings.MODEL_PATH}'")
        model = joblib.load(settings.MODEL_PATH)

    # wczytuje wymagane kolumny
    if os.path.isfile(settings.REQUIRED_COLUMNS_PATH):
        try:
            with open(settings.REQUIRED_COLUMNS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                required_columns[:] = (
                    data["columns"]
                    if isinstance(data, dict) and "columns" in data
                    else list(data)
                )
        except Exception:
            required_columns[:] = []
    else:
        required_columns[:] = []

    await database.connect()
    await init_db()

    try:
        yield
    finally:
        await database.disconnect()


# tworzy aplikację
app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    # zwraca status
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
async def predict(payload: Features):
    # sprawdza model
    if model is None:
        raise HTTPException(status_code=500, detail="model nie został załadowany")

    # buduje dataframe z payloadu
    x = pd.DataFrame([payload.data])

    # ustawia kolumny pod model
    if required_columns:
        x = x.reindex(columns=list(required_columns), fill_value=0)

    # robi predykcję
    try:
        pred = model.predict(x)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inferencja nie powiodła się: {e}")

    # rzutuje wynik
    try:
        pred_out: float | str = float(pred)
    except Exception:
        pred_out = str(pred)

    # zapisuje predykcję do bazy
    await save_prediction(payload.data, pred_out, MODEL_VERSION)

    # zwraca odpowiedź
    return Prediction(prediction=pred_out, model_version=MODEL_VERSION)


async def init_db():
    # wybiera schemat tabeli
    backend = database.url.scheme

    if backend.startswith("sqlite"):
        query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                payload TEXT,
                prediction REAL,
                model_version TEXT
            )
        """
    else:
        query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP,
                payload JSONB,
                prediction DOUBLE PRECISION,
                model_version TEXT
            )
        """

    # tworzy tabelę
    await database.execute(query=query)


async def save_prediction(
    payload: dict,
    prediction: float | str,
    model_version: str,
):
    # zapis pojedynczej predykcji
    backend = database.url.scheme
    is_sqlite = backend.startswith("sqlite")

    ts_value = dt.datetime.utcnow().isoformat() if is_sqlite else dt.datetime.utcnow()
    payload_value = json.dumps(payload) if is_sqlite else payload

    query = """
        INSERT INTO predictions(ts, payload, prediction, model_version)
        VALUES (:ts, :payload, :pred, :ver)
    """

    # zapisuje rekord
    await database.execute(
        query=query,
        values={
            "ts": ts_value,
            "payload": payload_value,
            "pred": float(prediction) if isinstance(prediction, (int, float)) else None,
            "ver": model_version,
        },
    )
