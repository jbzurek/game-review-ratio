from kedro.pipeline import Pipeline, node

from .nodes import (
    basic_clean,  # czyści i przygotowuje cechy
    choose_best_model,  # wybiera lepszy model
    evaluate_autogluon,  # ewaluacja autogluon
    evaluate_baseline,  # ewaluacja baseline
    load_raw,  # wczytuje surowe dane
    log_ag_metrics,  # zapisuje metryki autogluon
    log_baseline_metrics,  # zapisuje metryki baseline
    publish_production_artifact,  # publikuje model production do W&B
    save_production_model,  # zapisuje model produkcyjny
    split_data,  # dzieli dane na train i test
    train_autogluon,  # trenuje autogluon
    train_baseline,  # trenuje baseline
)


# tworzy pipeline kedro łączący cały proces modelowania
def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            # przygotowuje dane
            node(load_raw, "raw_data", "raw_df", name="load_raw"),
            node(
                basic_clean,
                ["raw_df", "params:clean", "params:target"],
                "clean_df",
                name="basic_clean",
            ),
            node(
                split_data,
                ["clean_df", "params:target", "params:split"],
                ["x_train", "x_test", "y_train", "y_test"],
                name="split_data",
            ),
            # trenuje i ocenia baseline
            node(
                train_baseline,
                ["x_train", "y_train", "params:model"],
                "baseline_model",
                name="train_baseline",
            ),
            node(
                evaluate_baseline,
                ["baseline_model", "x_test", "y_test"],
                "baseline_metrics_local",
                name="evaluate_baseline",
            ),
            node(
                log_baseline_metrics,
                "baseline_metrics_local",
                "baseline_metrics",
                name="log_baseline_metrics",
            ),
            # trenuje i ocenia autogluon
            node(
                train_autogluon,
                ["x_train", "y_train", "params:autogluon"],
                "ag_model",
                name="train_autogluon",
            ),
            node(
                evaluate_autogluon,
                ["ag_model", "x_test", "y_test"],
                "ag_metrics_local",
                name="evaluate_autogluon",
            ),
            node(
                log_ag_metrics,
                "ag_metrics_local",
                "ag_metrics",
                name="log_ag_metrics",
            ),
            # wybiera i zapisuje najlepszy model
            node(
                choose_best_model,
                ["ag_metrics_local", "baseline_metrics_local"],
                "best_model_name",
                name="choose_best_model",
            ),
            node(
                save_production_model,
                ["best_model_name"],
                "production_model_path",
                name="save_production_model",
            ),
            # publikuje model produkcyjny jako W&B Artifact z aliasem "production"
            node(
                publish_production_artifact,
                ["production_model_path", "best_model_name"],
                None,
                name="publish_production_artifact",
            ),
        ]
    )
