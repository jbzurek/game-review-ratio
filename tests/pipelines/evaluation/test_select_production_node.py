from src.gamereviewratio.pipelines.evaluation.nodes import choose_best_model


# testuje czy wybierany jest model z niższym rmse
def test_select_prefers_lower_rmse():
    ag = {"rmse": 0.12}
    base = {"rmse": 0.10}

    chosen = choose_best_model(ag, base)

    assert (
        chosen == "baseline_model"
    ), "powinien zostać wybrany model baseline, bo ma niższy RMSE"
