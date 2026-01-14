import pandas as pd
import pytest

from src.gamereviewratio.pipelines.evaluation.nodes import split_data


# testuje poprawność podziału danych na x i y
def test_split_returns_y_as_dataframe_and_no_target_in_x():
    df = pd.DataFrame(
        {"f1": [1, 2, 3, 4, 5], "pct_pos_total": [0.1, 0.2, 0.3, 0.4, 0.5]}
    )

    x_train, x_test, y_train, y_test = split_data(
        df, "pct_pos_total", {"test_size": 0.4, "random_state": 42}
    )

    assert (
        "pct_pos_total" not in x_train.columns
    ), "target nie powinien znaleźć się w cechach x"
    assert len(x_train) + len(x_test) == len(df), "suma x_train + x_test nie zgadza się"
    assert len(y_train) + len(y_test) == len(df), "suma y_train + y_test nie zgadza się"
    assert list(y_train.columns) == ["pct_pos_total"], "y_train ma złą strukturę"
    assert list(y_test.columns) == ["pct_pos_total"], "y_test ma złą strukturę"


# testuje obsługę błędu
def test_split_raises_if_target_missing():
    df = pd.DataFrame({"f1": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="Kolumna docelowa 'pct_pos_total' nie znajduje się w DataFrame",
    ):
        split_data(df, "pct_pos_total", {"test_size": 0.2, "random_state": 42})
