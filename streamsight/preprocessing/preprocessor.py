import logging
from typing import List, Literal

import pandas as pd
from tqdm.auto import tqdm

from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing.filter import Filter

tqdm.pandas()

logger = logging.getLogger(__name__)


class DataFramePreprocessor:
    """Preprocesses a pandas DataFrame into an InteractionMatrix object.

    The DataFramePreprocessor class allows the programmer to add filters for data
    preprocessing before transforming the data into an InteractionMatrix object.
    The preprocessor class after applying the filters, updates the item and user
    ID mappings into internal ID to reduce the computation load and allows for
    easy representation of the matrix.

    :param item_ix: Name of the column in which item identifiers are listed.
    :type item_ix: str
    :param user_ix: Name of the column in which user identifiers are listed.
    :type user_ix: str
    :param timestamp_ix: Name of the column in which timestamps are listed.
    :type timestamp_ix: str
    """

    def __init__(self, item_ix: str, user_ix: str, timestamp_ix: str):
        self._item_id_mapping = dict()
        self._user_id_mapping = dict()

        self.item_ix = item_ix
        self.user_ix = user_ix
        self.timestamp_ix = timestamp_ix

        self.filters: List[Filter] = []

    @property
    def item_id_mapping(self) -> pd.DataFrame:
        """Map from original item IDs to internal item IDs.

        Pandas DataFrame containing mapping from original item IDs to internal
        (consecutive) item IDs as columns.

        :return: DataFrame containing the mapping from original item IDs to internal
        :rtype: pd.DataFrame
        """
        return pd.DataFrame.from_records(
            list(self._item_id_mapping.items()),
            columns=[InteractionMatrix.ITEM_IX, self.item_ix],
        )

    @property
    def user_id_mapping(self) -> pd.DataFrame:
        """Map from original user IDs to internal user IDs.

        Pandas DataFrame containing mapping from original user IDs to internal
        (consecutive) user IDs as columns.

        :return: DataFrame containing the mapping from original item IDs to internal
        :rtype: pd.DataFrame
        """
        return pd.DataFrame.from_records(
            list(self._user_id_mapping.items()),
            columns=[InteractionMatrix.USER_IX, self.user_ix],
        )

    def add_filter(self, filter: Filter):
        """Add a preprocessing filter to be applied
        
        This filter will be applied before transforming to a
        :class:`InteractionMatrix` object.

        Filters are applied in order of addition, different orderings can lead to
        different results!

        :param filter: The filter to be applied
        :type filter: Filter
        """
        self.filters.append(filter)

    def _print_log_message(
        self,
        step: Literal["before", "after"],
        stage: Literal["preprocess", "filter"],
        df: pd.DataFrame,
    ):
        """Logging for change tracking.

        Prints a log message with the number of interactions, items and users
        in the DataFrame.

        :param step: To indicate if the log message is before or after the preprocessing
        :type step: Literal[&quot;before&quot;, &quot;after&quot;]
        :param stage: The current stage of the preprocessing
        :type stage: Literal[&quot;preprocess&quot;, &quot;filter&quot;]
        :param df: The dataframe being processed
        :type df: pd.DataFrame
        """
        logger.debug(f"\tinteractions {step} {stage}: {len(df.index)}")
        logger.debug(f"\titems {step} {stage}: {df[self.item_ix].nunique()}")
        logger.debug(f"\tusers {step} {stage}: {df[self.user_ix].nunique()}")

    def _update_id_mappings(self, df: pd.DataFrame) -> None:
        """Update the internal ID mappings for users and items.

        The internal ID mappings are updated to reduce the computation load and
        allow for easy representation of the matrix.

        :param df: DataFrame to update the ID mappings
        :type df: pd.DataFrame
        """
        # sort by timestamp to incrementally assign user and item ids by timestamp
        df.sort_values(by=[self.timestamp_ix], inplace=True, ignore_index=True)
        user_index = pd.CategoricalIndex(
            df[self.user_ix], categories=df[self.user_ix].unique()
        )
        self._user_id_mapping = dict(enumerate(user_index.drop_duplicates()))
        df[self.user_ix] = user_index.codes

        item_index = pd.CategoricalIndex(
            df[self.item_ix], categories=df[self.item_ix].unique()
        )
        self._item_id_mapping = dict(enumerate(item_index.drop_duplicates()))
        df[self.item_ix] = item_index.codes

    def process(self, df: pd.DataFrame) -> InteractionMatrix:
        self._print_log_message("before", "preprocess", df)

        for filter in self.filters:
            logger.debug(f"applying filter: {filter}")
            df = filter.apply(df)
            self._print_log_message("after", "filter", df)

        self._update_id_mappings(df)

        self._print_log_message("after", "preprocess", df)

        # Convert input data into internal data objects
        interaction_m = InteractionMatrix(
            df, self.item_ix, self.user_ix, self.timestamp_ix
        )

        return interaction_m
