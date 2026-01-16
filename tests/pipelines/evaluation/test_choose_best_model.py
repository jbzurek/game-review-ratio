from src.gamereviewratio.pipelines.evaluation import nodes as ds_nodes


def test_choose_best_model_picks_lower_rmse():
    ag_metrics = {"rmse": 1.0}
    base_metrics = {"rmse": 2.0}

    best = ds_nodes.choose_best_model(ag_metrics, base_metrics)
    assert best == "ag_model"


def test_choose_best_model_picks_baseline_when_better():
    ag_metrics = {"rmse": 3.0}
    base_metrics = {"rmse": 2.0}

    best = ds_nodes.choose_best_model(ag_metrics, base_metrics)
    assert best == "baseline_model"
