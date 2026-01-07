import pytest

from streamsight.matrix import InteractionMatrix
from test.conftest import test_dataframe, TIMESTAMP_IX, ITEM_IX, USER_IX

@pytest.fixture()
def matrix(test_dataframe):
    return InteractionMatrix(test_dataframe, ITEM_IX, USER_IX, TIMESTAMP_IX)
