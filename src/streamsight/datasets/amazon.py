import logging
from typing import ClassVar

import numpy as np
import pandas as pd

from streamsight.datasets.base import Dataset
from streamsight.datasets.config import (
    AmazonBookDatasetConfig,
    AmazonDatasetConfig,
    AmazonMovieDatasetConfig,
    AmazonMusicDatasetConfig,
    AmazonSubscriptionBoxesDatasetConfig,
)
from .metadata.amazon import (
    AmazonBookItemMetadata,
    AmazonMovieItemMetadata,
    AmazonMusicItemMetadata,
    AmazonSubscriptionBoxesItemMetadata,
)


logger = logging.getLogger(__name__)


class AmazonDataset(Dataset):
    """Base class for Amazon datasets.

    Other Amazon datasets should inherit from this class.
    """
    ITEM_METADATA = None

    config: ClassVar[AmazonDatasetConfig] = AmazonDatasetConfig()

    def _download_dataset(self) -> None:
        """Downloads the dataset.

        Downloads the csv file from the dataset URL and saves it to the file path.
        """
        if not self.config.dataset_url:
            raise ValueError(f"{self.name} does not have URL specified in config.")

        logger.debug(f"Downloading {self.name} dataset from {self.config.dataset_url}")
        self._fetch_remote(
            self.config.dataset_url,
            self.file_path,
        )

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        Transform the dataset downloaded to have integer user and item ids. This
        will be needed for representation in the interaction matrix.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        self.fetch_dataset()

        # Read JSONL in chunks and show progress per chunk. We import tqdm
        # locally to avoid global pandas monkeypatching (`tqdm.pandas()`).
        from tqdm.auto import tqdm

        chunksize = 100_000
        chunks = pd.read_json(
            self.file_path,
            dtype={
                self.config.item_ix: str,
                self.config.user_ix: str,
                self.config.timestamp_ix: np.int64,
                self.config.rating_ix: np.float32,
                self.config.helpful_vote_ix: np.int64,
            },
            lines=True,
            chunksize=chunksize,
        )
        df = pd.concat(
            [chunk for chunk in tqdm(chunks, desc="Reading JSONL", unit="chunk")], ignore_index=True
        )

        df = df[
            [
                self.config.item_ix,
                self.config.user_ix,
                self.config.timestamp_ix,
                self.config.rating_ix,
                self.config.helpful_vote_ix,
            ]
        ]

        # Convert nanosecond timestamps to seconds
        df[self.config.timestamp_ix] = df[self.config.timestamp_ix] // 1_000_000_000

        logger.debug(f"Loaded {len(df)} interactions")
        return df


class AmazonMusicDataset(AmazonDataset):
    """Handles Amazon Music dataset."""

    config: ClassVar[AmazonMusicDatasetConfig] = AmazonMusicDatasetConfig()

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        self.ITEM_METADATA = AmazonMusicItemMetadata(item_id_mapping=item_id_mapping).load()


class AmazonMovieDataset(AmazonDataset):
    """Handles Amazon Movie dataset."""

    config: ClassVar[AmazonMovieDatasetConfig] = AmazonMovieDatasetConfig()

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        self.ITEM_METADATA = AmazonMovieItemMetadata(item_id_mapping=item_id_mapping).load()


class AmazonSubscriptionBoxesDataset(AmazonDataset):
    """Handles Amazon Computer dataset."""

    config: ClassVar[AmazonSubscriptionBoxesDatasetConfig] = AmazonSubscriptionBoxesDatasetConfig()

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        self.ITEM_METADATA = AmazonSubscriptionBoxesItemMetadata(
            item_id_mapping=item_id_mapping
        ).load()


class AmazonBookDataset(AmazonDataset):
    """Handles Amazon Book dataset."""

    config: ClassVar[AmazonBookDatasetConfig] = AmazonBookDatasetConfig()

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        self.ITEM_METADATA = AmazonBookItemMetadata(item_id_mapping=item_id_mapping).load()
