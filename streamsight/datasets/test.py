import pandas as pd

from streamsight.datasets.base import Dataset


class TestDataset(Dataset):
    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "item_id"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    DEFAULT_FILENAME = "dummy_input.csv"

    def _download_dataset(self):
        pass

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::

            This does not apply any preprocessing, and returns the raw dataset.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        input_dict = {
            self.USER_IX:      [1, 2, 3, 1, 2, 2, 4, 3, 3, 4, 5, 5, 5],
            self.ITEM_IX:      [1, 1, 2, 3, 2, 3, 2, 1, 3, 3, 1, 2, 3],
            self.TIMESTAMP_IX: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10],
        }

        df = pd.DataFrame.from_dict(input_dict)
        return df