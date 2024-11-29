"""Data preprocessing module.

This module provides the DataFramePreprocessor class for converting pandas
DataFrames into InteractionMatrix objects with optional filtering.
"""

import logging
from typing import Literal

import pandas as pd

from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing.filter import Filter


logger = logging.getLogger(__name__)


class DataFramePreprocessor:
    """Preprocesses pandas DataFrames into InteractionMatrix objects.

    Allows adding filters for data preprocessing before transforming data into
    an InteractionMatrix object. After applying filters, updates item and user
    ID mappings to internal IDs to reduce computation load and enable easy
    matrix representation.

    Args:
        item_ix: Column name containing item identifiers.
        user_ix: Column name containing user identifiers.
        timestamp_ix: Column name containing timestamps.
    """

    def __init__(self, item_ix: str, user_ix: str, timestamp_ix: str) -> None:
        self._item_id_mapping = dict()
        self._user_id_mapping = dict()
        self.item_ix = item_ix
        self.user_ix = user_ix
        self.timestamp_ix = timestamp_ix
        self.filters: list[Filter] = []

    @property
    def item_id_mapping(self) -> pd.DataFrame:
        """Map from original item IDs to internal item IDs.

        Returns:
            DataFrame with columns for internal item IDs and original item IDs.
        """
        return pd.DataFrame.from_records(
            list(self._item_id_mapping.items()),
            columns=[InteractionMatrix.ITEM_IX, self.item_ix],
        )

    @property
    def user_id_mapping(self) -> pd.DataFrame:
        """Map from original user IDs to internal user IDs.

        Returns:
            DataFrame with columns for internal user IDs and original user IDs.
        """
        return pd.DataFrame.from_records(
            list(self._user_id_mapping.items()),
            columns=[InteractionMatrix.USER_IX, self.user_ix],
        )

    def add_filter(self, filter: Filter) -> None:
        """Add a preprocessing filter to be applied.

        The filter will be applied before transforming to an InteractionMatrix
        object. Filters are applied in order of addition and different orderings
        can lead to different results.

        Args:
            filter: The filter to be applied.
        """
        self.filters.append(filter)

    def _print_log_message(
        self,
        step: Literal["before", "after"],
        stage: Literal["preprocess", "filter"],
        df: pd.DataFrame,
    ) -> None:
        """Log preprocessing progress.

        Prints a log message with the number of interactions, items, and users
        in the DataFrame at the current stage.

        Args:
            step: Indicates whether log is before or after preprocessing.
            stage: Current stage of preprocessing (preprocess or filter).
            df: The DataFrame being processed.
        """
        logger.debug(f"\tinteractions {step} {stage}: {len(df.index)}")
        logger.debug(f"\titems {step} {stage}: {df[self.item_ix].nunique()}")
        logger.debug(f"\tusers {step} {stage}: {df[self.user_ix].nunique()}")

    def _update_id_mappings(self, df: pd.DataFrame) -> None:
        """Update internal ID mappings for users and items.

        Internal ID mappings are updated to reduce computation load and enable
        easy matrix representation. IDs are assigned by timestamp order.

        Args:
            df: DataFrame to update ID mappings for.
        """
        # Sort by timestamp to incrementally assign user and item ids by timestamp
        df.sort_values(by=[self.timestamp_ix], inplace=True, ignore_index=True)
        user_index = pd.CategoricalIndex(df[self.user_ix], categories=df[self.user_ix].unique())
        self._user_id_mapping = dict(enumerate(user_index.drop_duplicates()))
        df[self.user_ix] = user_index.codes

        item_index = pd.CategoricalIndex(df[self.item_ix], categories=df[self.item_ix].unique())
        self._item_id_mapping = dict(enumerate(item_index.drop_duplicates()))
        df[self.item_ix] = item_index.codes

    def process(self, df: pd.DataFrame) -> InteractionMatrix:
        """Process DataFrame through filters and convert to InteractionMatrix.

        Args:
            df: DataFrame to process.

        Returns:
            InteractionMatrix object created from processed DataFrame.
        """
        self._print_log_message("before", "preprocess", df)

        for filter_obj in self.filters:
            logger.debug(f"applying filter: {filter_obj}")
            df = filter_obj.apply(df)
            self._print_log_message("after", "filter", df)

        self._update_id_mappings(df)
        self._print_log_message("after", "preprocess", df)

        # Convert input data into internal data objects
        interaction_m = InteractionMatrix(df, self.item_ix, self.user_ix, self.timestamp_ix)
        return interaction_m
