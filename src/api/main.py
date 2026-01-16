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

# AutoGluon import (tylko jeśli model jest katalogiem)
from autogluon.tabular import TabularPredictor


# =========================
# Settings
# =========================

class Settings(BaseSettings):
    MODEL_PATH: str = "data/06_models/production_model"
    REQUIRED_COLUMNS_PATH: str = "data/06_models/required_columns.json"
    DATABASE_URL: str = "sqlite+aiosqlite:///./predictions.db"
    MODEL_VERSION: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
database = Database(settings.DATABASE_URL)

model: Any | None = None
model_type: str | None = None   # "autogluon" | "sklearn"
required_columns: list[str] = []


# =========================
# Utils
# =========================

def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_model_version() -> str:
    if settings.MODEL_VERSION:
        return settings.MODEL_VERSION

    p = Path(settings.MODEL_PATH)
    if p.is_file():
        return f"{p.name}:{_file_sha256(str(p))[:12]}"
    if p.is_dir():
        return f"{p.name}:{_file_sha256(str(p / 'learner.pkl'))[:12]}"

    return "unknown"


MODEL_VERSION = _resolve_model_version()


def _build_feature_row(req: "PredictRequest") -> pd.DataFrame:
    if not required_columns:
        raise RuntimeError("required_columns nie zostały załadowane")

    row = pd.DataFrame([{c: 0.0 for c in required_columns}], dtype=float)

    base = {
        "required_age": float(req.required_age),
        "price": float(req.price),
        "dlc_count": float(req.dlc_count),
        "windows": float(req.windows),
        "mac": float(req.mac),
        "linux": float(req.linux),
        "metacritic_score": float(req.metacritic_score),
        "achievements": float(req.achievements),
        "discount": float(req.discount),
        "release_year": float(req.release_year),
        "release_month": float(req.release_month),
    }

    for k, v in base.items():
        if k in row.columns:
            row.at[0, k] = v

    def set_mlb(prefix: str, values: list[str]) -> None:
        for val in values:
            col = f"{prefix}_{val}"
            if col in row.columns:
                row.at[0, col] = 1.0

    set_mlb("genres", req.genres)
    set_mlb("categories", req.categories)
    set_mlb("tags", req.tags)
    set_mlb("developers", req.developers)
    set_mlb("publishers", req.publishers)
    set_mlb("supported_languages", req.supported_languages)
    set_mlb("full_audio_languages", req.full_audio_languages)

    return row


# =========================
# Schemas
# =========================

class PredictRequest(BaseModel):
    required_age: int = Field(0, ge=0, le=99)
    price: float = Field(19.99, ge=0.0)
    dlc_count: int = Field(0, ge=0)

    windows: bool = True
    mac: bool = False
    linux: bool = False

    metacritic_score: int = Field(0, ge=0, le=100)
    achievements: int = Field(0, ge=0)
    discount: int = Field(0, ge=0, le=100)

    release_year: int = Field(2024, ge=1990, le=2035)
    release_month: int = Field(6, ge=1, le=12)

    genres: List[str] = []
    categories: List[str] = []
    tags: List[str] = []
    developers: List[str] = []
    publishers: List[str] = []
    supported_languages: List[str] = []
    full_audio_languages: List[str] = []


class Prediction(BaseModel):
    prediction: float
    model_version: str


# =========================
# Lifespan
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_type, required_columns

    model_path = Path(settings.MODEL_PATH)

    # ---- load model ----
    if model_path.is_dir():
        model = TabularPredictor.load(str(model_path))
        model_type = "autogluon"
    elif model_path.is_file():
        model = joblib.load(model_path)
        model_type = "sklearn"
    else:
        raise RuntimeError(f"nie znaleziono modelu w '{settings.MODEL_PATH}'")

    # ---- required columns ----
    if not os.path.isfile(settings.REQUIRED_COLUMNS_PATH):
        raise RuntimeError(
            f"nie znaleziono required_columns w '{settings.REQUIRED_COLUMNS_PATH}'"
        )

    with open(settings.REQUIRED_COLUMNS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        required_columns[:] = (
            data["columns"] if isinstance(data, dict) else list(data)
        )

    # ---- database ----
    await database.connect()
    await init_db()

    try:
        yield
    finally:
        await database.disconnect()


app = FastAPI(lifespan=lifespan)


# =========================
# Endpoints
# =========================

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model_type": model_type}


@app.get("/version")
async def version():
    return {"model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
async def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="model nie został załadowany")

    try:
        x = _build_feature_row(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"feature build failed: {e}")

    try:
        if model_type == "autogluon":
            pred = model.predict(x)
            pred_out = float(pred.iloc[0])
        else:
            pred = model.predict(x)
            pred_out = float(pred[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inferencja nie powiodła się: {e}")

    try:
        await save_prediction(payload.model_dump(), pred_out, MODEL_VERSION, database)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db write failed: {e}")

    return Prediction(prediction=pred_out, model_version=MODEL_VERSION)


# =========================
# DB helpers
# =========================

async def init_db():
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

    await database.execute(query=query)


async def save_prediction(payload: dict, prediction: float, model_version: str, database: Database):
    """
    Zapisuje predykcję do bazy danych.
    Działa dla SQLite (TEXT) i PostgreSQL (JSONB).
    """
    backend = database.url.scheme
    is_sqlite = backend.startswith("sqlite")

    ts_value = dt.datetime.utcnow().isoformat() if is_sqlite else dt.datetime.utcnow()

    # Zawsze serializujemy do JSON string
    payload_json = json.dumps(payload, ensure_ascii=False)

    if is_sqlite:
        query = """
            INSERT INTO predictions(ts, payload, prediction, model_version)
            VALUES (:ts, :payload, :pred, :ver)
        """
    else:
        # PostgreSQL: używamy CAST na JSONB
        query = """
            INSERT INTO predictions(ts, payload, prediction, model_version)
            VALUES (:ts, CAST(:payload AS JSONB), :pred, :ver)
        """

    await database.execute(
        query=query,
        values={
            "ts": ts_value,
            "payload": payload_json,
            "pred": float(prediction),
            "ver": model_version,
        },
    )
