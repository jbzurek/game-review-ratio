import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from pathlib import Path

from gamereviewratio.pipelines.evaluation.nodes import (
    evaluate_autogluon,
    train_baseline,
)


# testuje czy evaluate_autogluon zwraca słownik poprawnych metryk
def test_evaluate_autogluon_returns_correct_metrics():
    x_test = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_test = pd.DataFrame({"pct_pos_total": [0.5, 0.0, 1.0]})

    predictor = MagicMock()
    predictor.predict.return_value = np.zeros(len(x_test))

    metrics = evaluate_autogluon(predictor, x_test, y_test)

    assert isinstance(metrics, dict), "evaluate_autogluon powinno zwracać słownik"
    assert set(metrics.keys()) == {"rmse", "mae", "r2"}, "brakuje wymaganych metryk"

    assert metrics["rmse"] >= 0, "RMSE powinno być >= 0"
    assert metrics["mae"] >= 0, "MAE powinno być >= 0"
    assert metrics["r2"] <= 1, "R2 powinno być <= 1"
    assert metrics["r2"] > -10, "R2 nie powinno być ekstremalnie niskie"


# testuje czy trenowanie baseline tworzy katalog i zapisuje model
def test_train_baseline_creates_model_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    x_train = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y_train = pd.Series([0.1, 0.2, 0.3, 0.4])

    params = {"random_state": 42, "n_estimators": 10, "n_jobs": 1}

    train_baseline(x_train, y_train, params)

    models_dir = Path("data/06_models")
    model_file = models_dir / "baseline_model.pkl"

    assert models_dir.exists(), "katalog data/06_models powinien zostać utworzony"
    assert model_file.exists(), "plik baseline_model.pkl powinien zostać zapisany"
