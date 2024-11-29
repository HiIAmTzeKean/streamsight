import logging
import os
import time
from abc import ABC, abstractmethod
from typing import ClassVar

import httpx
import pandas as pd

from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing.filter import Filter, MinItemsPerUser, MinUsersPerItem
from streamsight.preprocessing.preprocessor import DataFramePreprocessor
from streamsight.utils.path import safe_dir
from .config import DatasetConfig


logger = logging.getLogger(__name__)


class DataFetcher(ABC):
    """Represents a abstract class to be used by Dataset or Metadata subclass.
    """

    config: ClassVar[DatasetConfig] = DatasetConfig()
    """Configuration for the dataset."""

    @property
    def name(self) -> str:
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def file_path(self) -> str:
        """File path of the dataset."""
        return os.path.join(self.config.default_base_path, self.config.default_filename)

    @property
    def processed_cache_path(self) -> str:
        """Path for cached processed data."""
        return os.path.join(
            self.config.default_base_path, f"{self.config.default_filename}.processed.parquet"
        )

    def fetch_dataset(self) -> None:
        """Check if dataset is present, if not download"""
        if os.path.exists(self.file_path):
            logger.debug("Data file is in memory and in dir specified.")
            return
        logger.debug(f"{self.name} dataset not found in {self.file_path}.")
        self._download_dataset()

    def fetch_dataset_force(self) -> None:
        """Force re-download of the dataset."""
        logger.debug(f"{self.name} force re-download of dataset.")
        self._download_dataset()

    def _fetch_remote(self, url: str, filename: str) -> str:
        """Fetch data from remote url and save locally (synchronous fallback).

        This function keeps the previous synchronous behaviour but uses
        `httpx.Client` to stream the response and write to disk. If you
        want async behaviour, use :meth:`_fetch_remote_async` instead.

        :param url: url to fetch data from
        :param filename: Path to save file to
        :return: The filename where data was saved
        """
        logger.debug(f"{self.name} will fetch dataset from remote url at {url}.")

        with httpx.Client(timeout=httpx.Timeout(60.0)) as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(filename, "wb") as fd:
                for chunk in resp.iter_bytes():
                    if chunk:
                        fd.write(chunk)

        return filename

    async def _fetch_remote_async(self, url: str, filename: str) -> str:
        """Asynchronously fetch data from a remote URL and save locally.

        Uses `httpx.AsyncClient` and streams the response to disk. Callers
        running inside an event loop should use this coroutine instead of
        the synchronous `_fetch_remote`.

        :param url: url to fetch data from
        :param filename: Path to save file to
        :return: The filename where data was saved
        """
        logger.debug(f"{self.name} will asynchronously fetch dataset from {url}.")

        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            # Write bytes as they arrive
            with open(filename, "wb") as fd:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        fd.write(chunk)

        return filename

    def _cache_processed_dataframe(self, df: pd.DataFrame) -> None:
        """Cache the processed DataFrame to disk.

        :param df: DataFrame to cache
        :type df: pd.DataFrame
        """
        logger.debug(f"Caching processed DataFrame to {self.processed_cache_path}")
        df.to_parquet(self.processed_cache_path)
        logger.debug("Processed DataFrame cached successfully.")

    @abstractmethod
    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        .. warning::
            This does not apply any preprocessing, and returns the raw dataset.

        :return: Interation with minimal columns of {user, item, timestamp}.
        :rtype: pd.DataFrame
        """
        raise NotImplementedError("Needs to be implemented")

    @abstractmethod
    def _download_dataset(self) -> None:
        """Downloads the dataset.

        Downloads the csv file from the dataset URL and saves it to the file path.
        """
        raise NotImplementedError("Needs to be implemented")


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

        if use_default_filters:
            for f in self._default_filters:
                self.add_filter(f)

        safe_dir(self.config.default_base_path)
        logger.debug(f"{self.name} is initialized.")

    @property
    def _default_filters(self) -> list[Filter]:
        """The default filters for all datasets

        Concrete classes can override this property to add more filters.

        :return: List of filters to be applied to the dataset
        :rtype: List[Filter]
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

    def add_filter(self, filter: Filter) -> None:
        """Add a filter to be applied when loading the data.

        Utilize :class:`DataFramePreprocessor` class to add filters to the
        dataset to load. The filter will be applied when the data is loaded into
        an :class:`InteractionMatrix` object when :meth:`load` is called.

        :param filter: Filter to be applied to the loaded DataFrame
                    processing to interaction matrix.
        :type filter: Filter
        """
        self.preprocessor.add_filter(filter)

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

        :param apply_filters: To apply the filters set and preprocessing,
            defaults to True
        :type apply_filters: bool, optional
        :return: Resulting interaction matrix
        :rtype: InteractionMatrix
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

        :param df: DataFrame to convert
        :type df: pd.DataFrame
        :return: InteractionMatrix object
        :rtype: InteractionMatrix
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
