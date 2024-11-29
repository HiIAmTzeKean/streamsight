import numpy as np
import pandas as pd

from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing import MinItemsPerUser


class TestMinItemsPerUser:
    def test_min_items_per_user_dataframe(self, test_dataframe: pd.DataFrame, session_vars) -> None:
        """Test the MinItemsPerUser filter on a DataFrame."""
        min_items = MinItemsPerUser(
            session_vars["MIN_ITEM_USER"],
            session_vars["ITEM_IX"],
            session_vars["USER_IX"],
        )
        filtered_data = min_items.apply(test_dataframe)
        assert all(filtered_data.value_counts(session_vars["USER_IX"]) >= session_vars["MIN_ITEM_USER"])

    def test_min_items_per_user_with_large_value(self, test_dataframe, session_vars):
        """When the minimum items per user is larger max number of interactions, the dataframe should be empty."""
        min_items = MinItemsPerUser(
            np.iinfo(np.int32).max,
            session_vars["ITEM_IX"],
            session_vars["USER_IX"],
        )
        filtered_data = min_items.apply(test_dataframe)
        assert len(filtered_data) == 0
        assert filtered_data.empty

    def test_min_items_per_user_with_zero_value(self, test_dataframe, session_vars):
        """When there is no minimum items per user, the dataframe should be the same as the original."""
        dataframe_copy = test_dataframe.copy()
        min_items = MinItemsPerUser(
            0,
            session_vars["ITEM_IX"],
            session_vars["USER_IX"],
        )
        filtered_data = min_items.apply(test_dataframe)
        assert len(filtered_data) == len(dataframe_copy)

    def test_min_items_per_user_dataset(self, test_dataset, test_dataframe, session_vars):
        test_dataset.add_filter(
            MinItemsPerUser(
                session_vars["MIN_ITEM_USER"],
                test_dataset.session_vars["ITEM_IX"],
                test_dataset.session_vars["USER_IX"],
            )
        )
        m = test_dataset.load()

        min_items = MinItemsPerUser(
            session_vars["MIN_ITEM_USER"],
            item_ix=session_vars["ITEM_IX"],
            user_ix=session_vars["USER_IX"],
        )
        filtered_data = min_items.apply(test_dataframe)
        filtered_data = filtered_data[[session_vars["TIMESTAMP_IX"], session_vars["USER_IX"], session_vars["ITEM_IX"]]]
        filtered_data.columns = ["ts", "uid", "iid"]
        # ID for load will start from 0
        filtered_data["uid"] = filtered_data["uid"] - 1
        filtered_data["iid"] = filtered_data["iid"] - 1

        assert m._df[["ts", "uid", "iid"]].values.tolist() == filtered_data.values.tolist()
        assert m._df.value_counts(InteractionMatrix.USER_IX).min() >= session_vars["MIN_ITEM_USER"]
