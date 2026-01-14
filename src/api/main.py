from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd
from databases import Database
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ścieżka do modelu
    MODEL_PATH: str = "data/06_models/production_model.pkl"

    # ścieżka do wymaganych kolumn
    REQUIRED_COLUMNS_PATH: str = "data/06_models/required_columns.json"

    # URL do bazy danych (docker compose: postgresql+asyncpg://app:app@db:5432/appdb)
    DATABASE_URL: str = "sqlite+aiosqlite:///./predictions.db"

    # wersja modelu
    MODEL_VERSION: str | None = None

    # klucz do Weights & Biases
    WANDB_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# tworzy połączenie do bazy
database = Database(settings.DATABASE_URL)

# trzyma model w pamięci
model: Any | None = None

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


def _parse_list(s: str | None) -> list[str]:
    # parsuje "a, b, c" -> ["a","b","c"]
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_feature_row(req: "PredictRequest") -> pd.DataFrame:
    # buduje wiersz cech zgodny z required_columns.json
    if not required_columns:
        raise RuntimeError("required_columns nie zostały załadowane")

    row = pd.DataFrame([{c: 0 for c in required_columns}])

    # cechy numeryczne/binarne
    base = {
        "required_age": int(req.required_age),
        "price": float(req.price),
        "dlc_count": int(req.dlc_count),
        "windows": int(bool(req.windows)),
        "mac": int(bool(req.mac)),
        "linux": int(bool(req.linux)),
        "metacritic_score": int(req.metacritic_score),
        "achievements": int(req.achievements),
        "discount": int(req.discount),
        "release_year": int(req.release_year),
        "release_month": int(req.release_month),
    }

    for k, v in base.items():
        if k in row.columns:
            row.at[0, k] = v

    def set_mlb(prefix: str, values: list[str]) -> None:
        for val in values:
            col = f"{prefix}_{val}"
            if col in row.columns:
                row.at[0, col] = 1

    set_mlb("genres", req.genres)
    set_mlb("categories", req.categories)
    set_mlb("tags", req.tags)
    set_mlb("developers", req.developers)
    set_mlb("publishers", req.publishers)
    set_mlb("supported_languages", req.supported_languages)
    set_mlb("full_audio_languages", req.full_audio_languages)

    return row


class PredictRequest(BaseModel):
    required_age: int = Field(default=0, ge=0, le=99)
    price: float = Field(default=19.99, ge=0.0)
    dlc_count: int = Field(default=0, ge=0)

    windows: bool = True
    mac: bool = False
    linux: bool = False

    metacritic_score: int = Field(default=0, ge=0, le=100)
    achievements: int = Field(default=0, ge=0)
    discount: int = Field(default=0, ge=0, le=100)

    release_year: int = Field(default=2024, ge=1990, le=2035)
    release_month: int = Field(default=6, ge=1, le=12)

    genres: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    developers: List[str] = Field(default_factory=list)
    publishers: List[str] = Field(default_factory=list)
    supported_languages: List[str] = Field(default_factory=list)
    full_audio_languages: List[str] = Field(default_factory=list)


class Prediction(BaseModel):
    # zwraca predykcję i wersję modelu
    prediction: float
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
        with open(settings.REQUIRED_COLUMNS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            cols = (
                data["columns"]
                if isinstance(data, dict) and "columns" in data
                else list(data)
            )
            required_columns[:] = list(cols)
    else:
        required_columns[:] = []

    await database.connect()
    await init_db()

    try:
        yield
    finally:
        await database.disconnect()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    # zwraca status
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
async def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="model nie został załadowany")

    # buduje dataframe pod model
    try:
        x = _build_feature_row(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"nie udało się zbudować cech: {e}")

    try:
        pred = model.predict(x)[0]
        pred_out = float(pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inferencja nie powiodła się: {e}")

    try:
        await save_prediction(payload.model_dump(), pred_out, MODEL_VERSION)
    except Exception:
        pass

    # zwraca odpowiedź
    return Prediction(prediction=pred_out, model_version=MODEL_VERSION)


async def init_db():
    backend = database.url.scheme

    if backend.startswith("sqlite"):
        query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                payload TEXT,
                prediction TEXT,
                model_version TEXT
            )
        """
    else:
        query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP,
                payload JSONB,
                prediction TEXT,
                model_version TEXT
            )
        """

    # tworzy tabelę
    await database.execute(query=query)


async def save_prediction(
    payload: dict,
    prediction: float,
    model_version: str,
):
    # zapis pojedynczej predykcji
    backend = database.url.scheme
    is_sqlite = backend.startswith("sqlite")

    ts_value = dt.datetime.utcnow().isoformat() if is_sqlite else dt.datetime.utcnow()
    payload_value = json.dumps(payload, ensure_ascii=False) if is_sqlite else payload

    query = """
        INSERT INTO predictions(ts, payload, prediction, model_version)
        VALUES (:ts, :payload, :pred, :ver)
    """

    await database.execute(
        query=query,
        values={
            "ts": ts_value,
            "payload": payload_value,
            "pred": str(prediction),
            "ver": model_version,
        },
    )
