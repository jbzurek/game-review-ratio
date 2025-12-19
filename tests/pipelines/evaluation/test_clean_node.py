import pandas as pd

from src.gamereviewratio.pipelines.evaluation.nodes import _parse_list_cell, basic_clean


# testuje parser list
def test_parse_list_cell_handles_list_and_csv_and_literal():
    assert _parse_list_cell(["a", "b"]) == ["a", "b"]
    assert _parse_list_cell("a,b") == ["a", "b"]
    assert _parse_list_cell("['x','y']") == ["x", "y"]
    assert _parse_list_cell(None) == []


# testuje czyszczenie danych
def test_basic_clean_threshold_removes_heavy_na():
    df = pd.DataFrame({"x": [1, None, None], "pct_pos_total": [0.1, 0.2, 0.3]})
    params = {
        "threshold_missing": 0.5,
        "bin_flag_cols": [],
        "platform_cols": [],
        "drop_cols": [],
        "mlb_cols": [],
        "top_n_labels": 10,
    }
    out = basic_clean(df, params, target="pct_pos_total")
    assert "x" not in out.columns
    assert "pct_pos_total" in out.columns
    assert not out.empty
