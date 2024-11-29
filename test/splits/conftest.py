import pandas as pd
import pytest

from streamsight.matrix import InteractionMatrix


@pytest.fixture
def matrix(test_dataframe: pd.DataFrame, session_vars: dict) -> InteractionMatrix:
    return InteractionMatrix(
        test_dataframe,
        session_vars["ITEM_IX"],
        session_vars["USER_IX"],
        session_vars["TIMESTAMP_IX"],
    )
