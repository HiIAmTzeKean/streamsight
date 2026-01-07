import pandas as pd
import pytest

from streamsight.datasets import Dataset, TestDataset

SEED = 42
BACKGROUND_T = 4
WINDOW_SIZE = 3
SEED = 42
N_SEQ_DATA=1
TIMESTAMP_IX = "timestamp"
ITEM_IX = "item_id"
USER_IX = "user_id"
MIN_ITEM_USER = 2


@pytest.fixture()
def test_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        TIMESTAMP_IX : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10],
        USER_IX :   [1, 2, 3, 1, 2, 2, 4, 3, 3, 4, 5, 5, 5],
        ITEM_IX :   [1, 1, 2, 3, 2, 3, 2, 1, 3, 3, 1, 2, 3],
        "rating": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })

@pytest.fixture()
def test_dataset() -> Dataset:
    dataset = TestDataset()
    return dataset