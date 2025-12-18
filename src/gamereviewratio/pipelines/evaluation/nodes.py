from __future__ import annotations

import ast
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from shutil import copyfile

import joblib
import numpy as np
import pandas as pd
import wandb
from autogluon.tabular import TabularPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# zwraca surowy dataframe bez zmian
def load_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df


# parsuje komórkę zawierającą listę lub tekst na listę stringów
def _parse_list_cell(x: Any) -> List[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x_str = x.strip()
        if not x_str:
            return []
        try:
            val = ast.literal_eval(x_str)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return list(val.keys())
        except Exception:
            pass
        return [s.strip() for s in x_str.split(",") if s.strip()]
    return []


# wykonuje podstawowe czyszczenie i inżynierię cech
def basic_clean(df: pd.DataFrame, clean: Dict[str, Any], target: str) -> pd.DataFrame:
    df = df.copy()
    threshold = float(clean.get("threshold_missing", 0.3))
    bin_flag_cols = list(clean.get("bin_flag_cols", []))
    platform_cols = list(clean.get("platform_cols", []))
    drop_cols = list(clean.get("drop_cols", []))
    mlb_cols = list(clean.get("mlb_cols", []))
    top_n = int(clean.get("top_n_labels", 50))

    to_drop = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].isnull().mean() > threshold:
            to_drop.append(col)
    if to_drop:
        df.drop(columns=to_drop, inplace=True, errors="ignore")

    for col in ["positive", "negative"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
        df["release_month"] = df["release_date"].dt.month
        df.drop(columns=["release_date"], inplace=True)

    for col in bin_flag_cols:
        if col in df.columns:
            df[f"has_{col}"] = df[col].notnull().astype(int)
            df.drop(columns=[col], inplace=True)

    for col in platform_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    for col in mlb_cols:
        if col not in df.columns:
            continue
        series = df[col].apply(_parse_list_cell)
        counts: Dict[str, int] = {}
        for lst in series.dropna():
            for lab in lst:
                counts[lab] = counts.get(lab, 0) + 1
        top_labels = sorted(counts, key=counts.get, reverse=True)[:top_n]
        top_set = set(top_labels)
        if not top_set:
            df.drop(columns=[col], inplace=True)
            continue
        series_top = series.apply(lambda lst: [x for x in lst if x in top_set])
        mlb = MultiLabelBinarizer(classes=sorted(top_set))
        encoded = pd.DataFrame(
            mlb.fit_transform(series_top),
            columns=[f"{col}_{c}" for c in mlb.classes_],
            index=df.index,
        )
        df = pd.concat([df.drop(columns=[col]), encoded], axis=1)

    return df


# dzieli dane na zbiory train i test oraz enkoduje cechy
def split_data(
    df: pd.DataFrame,
    target: str,
    split: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = float(split.get("test_size", 0.2))
    random_state = int(split.get("random_state", 42))

    if target not in df.columns:
        raise ValueError(f"Kolumna docelowa '{target}' nie znajduje się w DataFrame")

    y = pd.to_numeric(df[target], errors="coerce")
    x = df.drop(columns=[target])
    x = pd.get_dummies(x, drop_first=True)

    mask = y.notnull()
    x = x.loc[mask]
    y = y.loc[mask]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    return (
        x_train,
        x_test,
        y_train.to_frame(name=target),
        y_test.to_frame(name=target),
    )


# trenuje model baseline random forest i zapisuje go do pliku baseline_model.pkl
def train_baseline(
    x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, model: dict
) -> RandomForestRegressor:
    wandb.init(project="gamereviewratio", job_type="train", reinit=True, config=model)

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    y_train = y_train.to_numpy().ravel()

    params = {
        "random_state": model.get("random_state", 42),
        "n_estimators": model.get("n_estimators", 200),
        "n_jobs": model.get("n_jobs", -1),
    }

    mdl = RandomForestRegressor(**params)
    start = time.time()
    mdl.fit(x_train, y_train)
    train_time_s = time.time() - start

    path = Path("data/06_models/baseline_model.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, path)

    try:
        wandb.log({"train_time_s": train_time_s})
    except Exception:
        pass

    try:
        art = wandb.Artifact("baseline_model", type="model")
        art.add_file(str(path))
        wandb.log_artifact(art, aliases=["candidate"])
    except Exception:
        pass

    return mdl


# liczy metryki dla modelu baseline i loguje do wandb
def evaluate_baseline(
    mdl: RandomForestRegressor, x_test: pd.DataFrame, y_test: pd.DataFrame | pd.Series
) -> dict:
    if isinstance(y_test, pd.DataFrame):
        y_true = y_test.iloc[:, 0]
    else:
        y_true = y_test

    y_true = pd.to_numeric(y_true, errors="coerce")
    mask = y_true.notnull()
    y_true = y_true[mask]
    x_eval = x_test.loc[mask]

    pred = mdl.predict(x_eval)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))

    metrics = {"rmse": rmse}

    try:
        wandb.log(metrics)
    except Exception:
        pass

    try:
        wandb.finish()
    except Exception:
        pass

    return metrics


# trenuje model autogluon tabular i zapisuje go jako ag_model.pkl
def train_autogluon(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    ag_params: dict,
) -> TabularPredictor:
    label = ag_params.get("label")
    problem_type = ag_params.get("problem_type", "regression")
    eval_metric = ag_params.get("eval_metric", "rmse")
    time_limit = int(ag_params.get("time_limit", 60))
    presets = ag_params.get("presets", "medium_quality_faster_train")
    random_state = int(ag_params.get("random_state", 42))

    save_dir = Path("data/06_models/AutogluonModels")
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(y_train, pd.DataFrame):
        y_series = y_train.iloc[:, 0]
    else:
        y_series = y_train

    train_df = x_train.copy()
    train_df[label] = pd.to_numeric(y_series, errors="coerce")
    train_df = train_df.dropna(subset=[label])

    try:
        req_cols_path = Path("data/06_models/required_columns.json")
        req_cols_path.parent.mkdir(parents=True, exist_ok=True)
        cols_list = list((train_df.drop(columns=[label]).columns))
        with open(req_cols_path, "w", encoding="utf-8") as f:
            json.dump({"columns": cols_list}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    random.seed(random_state)
    np.random.seed(random_state)

    wandb_config = {
        "label": label,
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "time_limit": time_limit,
        "presets": presets,
        "random_state": random_state,
    }

    wandb_config.update(
        {f"feature_{col}": str(train_df[col].dtype) for col in x_train.columns}
    )

    wandb.init(
        project="gamereviewratio", job_type="ag-train", reinit=True, config=wandb_config
    )

    start = time.time()

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=str(save_dir),
        verbosity=2,
    ).fit(
        train_data=train_df,
        time_limit=time_limit,
        presets=presets,
    )

    train_time_s = time.time() - start

    pkl_path = Path("data/06_models/ag_model.pkl")
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, pkl_path)

    try:
        wandb.log({"train_time_s": train_time_s})
    except Exception:
        pass

    try:
        art = wandb.Artifact("ag_model", type="model")
        art.add_file(str(pkl_path))
        wandb.log_artifact(art, aliases=["candidate"])
    except Exception:
        pass

    return predictor


# liczy metryki dla modelu autogluon i loguje do wandb
def evaluate_autogluon(
    predictor: TabularPredictor,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame | pd.Series,
) -> Dict[str, float]:
    if isinstance(y_test, pd.DataFrame):
        y_true = y_test.iloc[:, 0]
    else:
        y_true = y_test

    y_true = pd.to_numeric(y_true, errors="coerce")
    mask = y_true.notnull()
    y_true = y_true[mask]
    x_eval = x_test.loc[mask]

    pred = predictor.predict(x_eval)

    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    r2 = float(r2_score(y_true, pred))

    metrics: Dict[str, float] = {"rmse": rmse, "mae": mae, "r2": r2}

    try:
        wandb.log(metrics)
    except Exception:
        pass

    try:
        wandb.finish()
    except Exception:
        pass

    return metrics


# zwraca metryki autogluon bez zmian
def log_ag_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return metrics


# zwraca metryki baseline bez zmian
def log_baseline_metrics(metrics: dict) -> dict:
    return metrics


# wybiera najlepszy model na podstawie rmse
def choose_best_model(ag_metrics: dict, baseline_metrics: dict) -> str:
    ag_rmse = ag_metrics.get("rmse", float("inf"))
    base_rmse = baseline_metrics.get("rmse", float("inf"))

    if ag_rmse <= base_rmse:
        return "ag_model"
    else:
        return "baseline_model"


# zapisuje lepszy model jako production_model.pkl
def save_production_model(best_model_name: str) -> str:
    if best_model_name == "ag_model":
        src = Path("data/06_models/ag_model.pkl")
    elif best_model_name == "baseline_model":
        src = Path("data/06_models/baseline_model.pkl")
    else:
        raise ValueError(f"Nieznana nazwa najlepszego modelu: {best_model_name!r}")

    dst = Path("data/06_models/production_model.pkl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    copyfile(src, dst)

    # zwraca ścieżkę
    return str(dst)
