import pandas as pd
import pytest

from src.gamereviewratio.pipelines.evaluation import nodes as ds_nodes


def test_split_data_shapes_ok():
    df = pd.DataFrame(
        {
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
            "windows": [1, 0, 1, 0, 1],
            "target": [10, 20, 30, 40, 50],
        }
    )

    x_train, x_test, y_train, y_test = ds_nodes.split_data(
        df,
        target="target",
        split={"test_size": 0.4, "random_state": 42},
    )

    assert len(x_train) + len(x_test) == 5
    assert len(y_train) + len(y_test) == 5
    assert "target" not in x_train.columns


def test_split_data_raises_on_missing_target():
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ValueError):
        ds_nodes.split_data(
            df, target="target", split={"test_size": 0.2, "random_state": 42}
        )
