import numpy as np
import pandas as pd

from src.gamereviewratio.pipelines.evaluation import nodes as ds_nodes


class DummyPredictor:
    def predict(self, X):
        return np.zeros(len(X))


def test_evaluate_autogluon_returns_expected_keys():
    predictor = DummyPredictor()

    x_test = pd.DataFrame({"a": [1, 2, 3]})
    y_test = pd.Series([0.0, 0.0, 0.0])

    metrics = ds_nodes.evaluate_autogluon(predictor, x_test, y_test)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert isinstance(metrics["rmse"], float)
