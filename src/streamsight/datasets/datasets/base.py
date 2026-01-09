import logging
import os
import time
from abc import abstractmethod
from datetime import datetime
from typing import ClassVar

import pandas as pd

from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing.filter import Filter, MinItemsPerUser, MinUsersPerItem
from streamsight.preprocessing.preprocessor import DataFramePreprocessor
from streamsight.utils.path import safe_dir
from ..base import DataFetcher
from ..config import DatasetConfig


logger = logging.getLogger(__name__)


class Dataset(DataFetcher):
    """Represents a collaborative filtering dataset.

    Dataset must minimally contain user, item and timestamp columns for the
    other modules to work.

    Assumption
    ===========
    User/item ID increments in the order of time. This is an assumption that will
    be made for the purposes of splitting the dataset and eventually passing
    the dataset to the model. The ID incrementing in the order of time allows us
    to set the shape of the currently known user and item matrix allowing easier
    manipulation of the data by the evaluator.

    :param filename: Name of the file, if no name is provided the dataset default will be used if known.
        If the dataset does not have a default filename, a ValueError will be raised.
    :type filename: str, optional
    :param base_path: The base_path to the data directory.
        Defaults to `data`
    :type base_path: str, optional
    :param use_default_filters: If True, the default filters will be applied to the dataset.
        Defaults to False.
    :type use_default_filters: bool, optional
    """

    config: ClassVar[DatasetConfig] = DatasetConfig()
    """Configuration for the dataset."""

    def __init__(
        self,
        use_default_filters: bool = False,  # noqa: FBT001, FBT002
        fetch_metadata: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        if not self.config.user_ix or not self.config.item_ix or not self.config.timestamp_ix:
            raise AttributeError("user_ix, item_ix or timestamp_ix not set in config.")

        logger.debug(
            f"{self.name} being initialized with '{self.config.default_base_path}' as the base path."
        )

        if not self.config.default_filename:
            raise ValueError("No filename specified, and no default known.")

        self.fetch_metadata = fetch_metadata
        self.preprocessor = DataFramePreprocessor(
            self.config.item_ix, self.config.user_ix, self.config.timestamp_ix
        )
        self._timestamp_min: int | None = None
        self._timestamp_max: int | None = None

        if use_default_filters:
            for f in self._default_filters:
                self.add_filter(f)

        safe_dir(self.config.default_base_path)
        logger.debug(f"{self.name} is initialized.")

    @property
    def _default_filters(self) -> list[Filter]:
        """The default filters for all datasets

        Concrete classes can override this property to add more filters.

        Returns:
            List of filters to be applied to the dataset.
        """
        if not self.config.user_ix or not self.config.item_ix:
            raise AttributeError("config.user_ix or config.item_ix not set.")

        filters: list[Filter] = []
        filters.append(
            MinItemsPerUser(
                min_items_per_user=3,
                item_ix=self.config.item_ix,
                user_ix=self.config.user_ix,
            )
        )
        filters.append(
            MinUsersPerItem(
                min_users_per_item=3,
                item_ix=self.config.item_ix,
                user_ix=self.config.user_ix,
            )
        )
        return filters

    @property
    def timestamp_min(self) -> int:
        """Minimum timestamp in the dataset.

        Returns:
            Minimum timestamp in the dataset.
        """
        if self._timestamp_min is None:
            raise RuntimeError("timestamp_min can only be accessed after load() has been called.")
        return self._timestamp_min

    @property
    def timestamp_max(self) -> int:
        """Maximum timestamp in the dataset.

        Returns:
            Maximum timestamp in the dataset.
        """
        if self._timestamp_max is None:
            raise RuntimeError("timestamp_max can only be accessed after load() has been called.")
        return self._timestamp_max

    def get_timestamp_range_in_epoch(self) -> tuple[int, int]:
        """Get the minimum and maximum timestamps in the dataset.

        Returns:
            A tuple of (min_timestamp, max_timestamp).

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        return self.timestamp_min, self.timestamp_max

    def get_timestamp_range_in_datetime(self) -> tuple[datetime, datetime]:
        """Get the minimum and maximum timestamps in the dataset.

        Returns:
            A tuple of (min_timestamp, max_timestamp).

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        min_dt = datetime.fromtimestamp(self.timestamp_min)
        max_dt = datetime.fromtimestamp(self.timestamp_max)
        return min_dt, max_dt

    def add_filter(self, filter_: Filter) -> None:
        """Add a filter to be applied when loading the data.

        Utilize :class:`DataFramePreprocessor` class to add filters to the
        dataset to load. The filter will be applied when the data is loaded into
        an :class:`InteractionMatrix` object when :meth:`load` is called.

        :param filter_: Filter to be applied to the loaded DataFrame
                    processing to interaction matrix.
        :type filter_: Filter
        """
        self.preprocessor.add_filter(filter_)

    def _load_dataframe_from_cache(self) -> pd.DataFrame:
        if not os.path.exists(self.processed_cache_path):
            raise FileNotFoundError("Processed cache file not found.")
        logger.info(f"Loading from cache: {self.processed_cache_path}")
        df = pd.read_parquet(self.processed_cache_path)
        return df

    def load(self, apply_filters: bool = True, use_cache: bool = True) -> InteractionMatrix:
        """Loads data into an InteractionMatrix object.

        Data is loaded into a DataFrame using the :func:`_load_dataframe` function.
        Resulting DataFrame is parsed into an :class:`InteractionMatrix` object. If
        :data:`apply_filters` is set to True, the filters set will be applied to the
        dataset and mapping of user and item ids will be done. This is advised
        even if there is no filter set, as it will ensure that the user and item
        ids are incrementing in the order of time.

        Args:
            apply_filters: To apply the filters set and preprocessing,
                defaults to True
            use_cache: Whether to use cached processed data, defaults to True

        Returns:
            Resulting interaction matrix.
        """
        logger.info(f"{self.name} is loading dataset...")
        start = time.time()
        try:
            df = self._load_dataframe_from_cache() if use_cache else self._load_dataframe()
        except FileNotFoundError:
            logger.warning("Processed cache not found, loading raw dataframe.")
            df = self._load_dataframe()
            self._cache_processed_dataframe(df)
        if apply_filters:
            logger.debug(f"{self.name} applying filters set.")
            im = self.preprocessor.process(df)
        else:
            im = self._dataframe_to_matrix(df)
            logger.warning(
                "No filters applied, user and item ids may not be incrementing in the order of time. "
                "Classes that use this dataset may not work as expected."
            )
        self._timestamp_min = int(df[self.config.timestamp_ix].min())
        self._timestamp_max = int(df[self.config.timestamp_ix].max())

        if self.fetch_metadata:
            user_id_mapping, item_id_mapping = (
                self.preprocessor.user_id_mapping,
                self.preprocessor.item_id_mapping,
            )
            self._fetch_dataset_metadata(
                user_id_mapping=user_id_mapping, item_id_mapping=item_id_mapping
            )

        end = time.time()
        logger.info(f"{self.name} dataset loaded - Took {end - start:.3}s")
        return im

    def _dataframe_to_matrix(self, df: pd.DataFrame) -> InteractionMatrix:
        """Converts a DataFrame to an InteractionMatrix.

        Args:
            df: DataFrame to convert

        Returns:
            InteractionMatrix object.
        """
        if not self.config.user_ix or not self.config.item_ix or not self.config.timestamp_ix:
            raise AttributeError("config.user_ix, config.item_ix or config.timestamp_ix not set.")
        return InteractionMatrix(
            df,
            user_ix=self.config.user_ix,
            item_ix=self.config.item_ix,
            timestamp_ix=self.config.timestamp_ix,
        )

    @abstractmethod
    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        """Fetch metadata for the dataset.

        Fetch metadata for the dataset, if available.
        """
        raise NotImplementedError("Needs to be implemented")
