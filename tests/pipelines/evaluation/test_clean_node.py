import pandas as pd

from src.gamereviewratio.pipelines.evaluation import nodes as ds_nodes


def test_basic_clean_drops_release_date_and_creates_time_features():
    df = pd.DataFrame(
        {
            "release_date": ["2024-06-10"],
            "price": [19.99],
            "target": [10.0],
        }
    )

    clean_params = {
        "threshold_missing": 0.9,
        "bin_flag_cols": [],
        "platform_cols": [],
        "drop_cols": [],
        "mlb_cols": [],
        "top_n_labels": 10,
    }

    out = ds_nodes.basic_clean(df, clean_params, target="target")

    assert "release_date" not in out.columns
    assert "release_year" in out.columns
    assert "release_month" in out.columns
    assert "target" in out.columns
