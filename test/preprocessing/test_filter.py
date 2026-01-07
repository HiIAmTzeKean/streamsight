import numpy as np
from streamsight.preprocessing import MinItemsPerUser
from streamsight.matrix import InteractionMatrix
from test.conftest import TIMESTAMP_IX, ITEM_IX, USER_IX, MIN_ITEM_USER

class TestMinItemsPerUser:
    def test_min_items_per_user_dataframe(self, test_dataframe):
        """Test the MinItemsPerUser filter on a DataFrame."""
        min_items = MinItemsPerUser(MIN_ITEM_USER, item_ix=ITEM_IX, user_ix=USER_IX)
        filtered_data = min_items.apply(test_dataframe)
        assert all(filtered_data.value_counts(USER_IX) >= MIN_ITEM_USER)

    def test_min_items_per_user_with_large_value(self, test_dataframe):
        """When the minimum items per user is larger max number of interactions, the dataframe should be empty."""
        min_items = MinItemsPerUser(np.iinfo(np.int32).max, item_ix=ITEM_IX, user_ix=USER_IX)
        filtered_data = min_items.apply(test_dataframe)
        assert len(filtered_data) == 0
        assert filtered_data.empty
        
    def test_min_items_per_user_with_zero_value(self, test_dataframe):
        """When there is no minimum items per user, the dataframe should be the same as the original."""
        dataframe_copy = test_dataframe.copy()
        min_items = MinItemsPerUser(0, item_ix=ITEM_IX, user_ix=USER_IX)
        filtered_data = min_items.apply(test_dataframe)
        assert len(filtered_data) == len(dataframe_copy)

    def test_min_items_per_user_dataset(self, test_dataset, test_dataframe):
        test_dataset.add_filter(MinItemsPerUser(MIN_ITEM_USER, item_ix=test_dataset.ITEM_IX, user_ix=test_dataset.USER_IX))
        m = test_dataset.load()
        
        min_items = MinItemsPerUser(MIN_ITEM_USER, item_ix=ITEM_IX, user_ix=USER_IX)
        filtered_data = min_items.apply(test_dataframe)
        filtered_data = filtered_data[[TIMESTAMP_IX, USER_IX, ITEM_IX]]
        filtered_data.columns = ["ts","uid","iid"]
        # ID for load will start from 0
        filtered_data["uid"] = filtered_data["uid"] - 1
        filtered_data["iid"] = filtered_data["iid"] - 1
        
        assert m._df[["ts","uid","iid"]].values.tolist() == filtered_data.values.tolist()
        assert m._df.value_counts(InteractionMatrix.USER_IX).min() >= MIN_ITEM_USER
    