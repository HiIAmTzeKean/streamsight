import pytest
from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing import DataFramePreprocessor, MinItemsPerUser
from test.conftest import test_dataset, test_dataframe, MIN_ITEM_USER


@pytest.fixture()
def preprocessor(test_dataset):
    return DataFramePreprocessor(
        test_dataset.ITEM_IX, test_dataset.USER_IX, test_dataset.TIMESTAMP_IX
    )


@pytest.fixture()
def filter(test_dataset):
    return MinItemsPerUser(3, test_dataset.ITEM_IX, test_dataset.USER_IX)


class TestDataFramePreprocessor:
    def test_add_filter(self, preprocessor, filter):
        preprocessor.add_filter(filter)

    def test_process_without_filter(self, preprocessor, test_dataframe):
        result = preprocessor.process(test_dataframe)
        assert preprocessor.item_id_mapping is not None
        assert preprocessor.item_id_mapping.to_dict() == {
            "iid": {0: 0, 1: 1, 2: 2},
            "item_id": {0: 1, 1: 2, 2: 3},
        }

        assert preprocessor.user_id_mapping is not None
        assert preprocessor.user_id_mapping.to_dict() == {
            "uid": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            "user_id": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        }
        assert type(result) == InteractionMatrix

    def test_process_with_filter(self, preprocessor, test_dataframe, filter):
        preprocessor.add_filter(filter)
        result = preprocessor.process(test_dataframe)
        assert preprocessor.item_id_mapping is not None
        assert preprocessor.item_id_mapping.to_dict() == {
            "iid": {0: 0, 1: 1, 2: 2},
            "item_id": {0: 1, 1: 2, 2: 3},
        }

        assert preprocessor.user_id_mapping is not None
        assert preprocessor.user_id_mapping.to_dict() == {
            "uid": {0: 0, 1: 1, 2: 2},
            "user_id": {0: 2, 1: 3, 2: 5},
        }
        assert type(result) == InteractionMatrix
