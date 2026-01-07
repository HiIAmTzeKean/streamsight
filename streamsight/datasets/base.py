import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve
from warnings import warn

import pandas as pd

from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing.filter import (Filter, MinItemsPerUser,
                                              MinUsersPerItem)
from streamsight.preprocessing.preprocessor import DataFramePreprocessor
from streamsight.utils import ProgressBar

logger = logging.getLogger(__name__)

class Dataset(ABC):
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

    USER_IX = None
    """Name of the column in the DataFrame with user identifiers"""
    ITEM_IX = None
    """Name of the column in the DataFrame with item identifiers"""
    TIMESTAMP_IX = None
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""

    DEFAULT_FILENAME = None
    """Default filename that will be used if it is not specified by the user."""

    DEFAULT_BASE_PATH = "data"
    """Default base path where the dataset will be stored."""

    def __init__(
        self, filename: Optional[str] = None, base_path: Optional[str] = None, use_default_filters=False
    ):
        if not self.USER_IX or not self.ITEM_IX or not self.TIMESTAMP_IX:
            raise AttributeError("USER_IX, ITEM_IX or TIMESTAMP_IX not set.")

        self.base_path = base_path if base_path else self.DEFAULT_BASE_PATH
        logger.debug(
            f"{self.name} being initialized with '{self.base_path}' as the base path."
        )

        self.filename = filename if filename else self.DEFAULT_FILENAME
        if not self.filename:
            raise ValueError("No filename specified, and no default known.")
        self.preprocessor = DataFramePreprocessor(
            self.ITEM_IX, self.USER_IX, self.TIMESTAMP_IX
        )
        
        if use_default_filters:
            for f in self._default_filters:
                self.add_filter(f)

        self._check_safe()
        logger.debug(f"{self.name} is initialized.")

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__
    
    @property
    def _default_filters(self) -> List[Filter]:
        """The default filters for all datasets
        
        Concrete classes can override this property to add more filters.
        
        :return: List of filters to be applied to the dataset
        :rtype: List[Filter]
        """
        return [
            MinItemsPerUser(3, self.ITEM_IX, self.USER_IX),
            MinUsersPerItem(3, self.ITEM_IX, self.USER_IX),
        ]

    @property
    def file_path(self) -> str:
        """File path of the dataset."""
        return os.path.join(self.base_path, self.filename) # type: ignore

    def _check_safe(self):
        """Check if the directory is safe. If directory does not exit, create it."""
        p = Path(self.base_path)
        p.mkdir(exist_ok=True)

    def fetch_dataset(self, force=False) -> None:
        """Check if dataset is present, if not download

        :param force: If True, dataset will be downloaded,
                even if the file already exists.
                Defaults to False.
        :type force: bool, optional
        """
        if not os.path.exists(self.file_path) or force:
            logger.debug(f"{self.name} dataset not found in {self.file_path}.")
            self._download_dataset()
        logger.debug(f"Data file is in memory and in dir specified.")

    def add_filter(self, filter: Filter):
        """Add a filter to be applied when loading the data.

        Utilize :class:`DataFramePreprocessor` class to add filters to the
        dataset to load. The filter will be applied when the data is loaded into
        an :class:`InteractionMatrix` object when :meth:`load` is called.
        
        :param filter: Filter to be applied to the loaded DataFrame
                    processing to interaction matrix.
        :type filter: Filter
        """
        self.preprocessor.add_filter(filter)

    def load(self,apply_filters=True) -> InteractionMatrix:
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
        df = self._load_dataframe()
        if apply_filters:
            logger.debug(f"{self.name} applying filters set.")
            im = self.preprocessor.process(df)
        else:
            im = self._dataframe_to_matrix(df)
            warn("No filters applied, user and item ids may not be incrementing in the order of time. Classes that use this dataset may not work as expected.")

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
        if not self.USER_IX or not self.ITEM_IX or not self.TIMESTAMP_IX:
            raise AttributeError("USER_IX, ITEM_IX or TIMESTAMP_IX not set.")
        return InteractionMatrix(
            df,
            user_ix=self.USER_IX,
            item_ix=self.ITEM_IX,
            timestamp_ix=self.TIMESTAMP_IX,
        )

    def _fetch_remote(self, url: str, filename: str) -> str:
        """Fetch data from remote url and save locally

        :param url: url to fetch data from
        :type url: str
        :param filename: Path to save file to
        :type filename: str
        :return: The filename where data was saved
        :rtype: str
        """
        logger.debug(f"{self.name} will fetch dataset from remote url at {url}.")
        urlretrieve(url, filename, ProgressBar())
        return filename

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
    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the csv file from the dataset URL and saves it to the file path.
        """
        raise NotImplementedError("Needs to be implemented")
