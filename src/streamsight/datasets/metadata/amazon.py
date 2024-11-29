import logging
import os
from typing import ClassVar

import pandas as pd

from streamsight.datasets.config import (
    AmazonBooksItemMetadataConfig,
    AmazonDigitalMusicItemMetadataConfig,
    AmazonItemMetadataConfig,
    AmazonMoviesAndTVItemMetadataConfig,
    AmazonSubscriptionBoxesItemMetadataConfig,
)
from .base import Metadata


logger = logging.getLogger(__name__)


class AmazonItemMetadata(Metadata):
    config: ClassVar[AmazonItemMetadataConfig] = AmazonItemMetadataConfig()

    def __init__(self, item_id_mapping: pd.DataFrame) -> None:
        super().__init__()
        self.item_id_mapping = item_id_mapping

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        df = pd.read_json(
            self.file_path,  # Ensure file_path contains the JSONL file path
            dtype=self.config.dtype_dict,
            lines=True,  # Required for JSONL format
        )

        item_id_to_iid = dict(zip(self.item_id_mapping[self.config.item_ix], self.item_id_mapping["iid"]))

        # Map config.item_ix in metadata_df using the optimized function
        df[self.config.item_ix] = df[self.config.item_ix].map(lambda x: item_id_to_iid.get(x, x))

        return df

    def _download_dataset(self) -> None:
        """Downloads the metadata for the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        if not self.config.dataset_url:
            raise ValueError(f"{self.name} does not have URL specified.")

        self._fetch_remote(
            self.config.dataset_url, os.path.join(self.base_path, f"{self.config.remote_filename}")
        )


class AmazonMusicItemMetadata(AmazonItemMetadata):
    config: ClassVar[AmazonDigitalMusicItemMetadataConfig] = AmazonDigitalMusicItemMetadataConfig()


class AmazonMovieItemMetadata(AmazonItemMetadata):
    config: ClassVar[AmazonMoviesAndTVItemMetadataConfig] = AmazonMoviesAndTVItemMetadataConfig()


class AmazonSubscriptionBoxesItemMetadata(AmazonItemMetadata):
    config: ClassVar[AmazonSubscriptionBoxesItemMetadataConfig] = AmazonSubscriptionBoxesItemMetadataConfig()


class AmazonBookItemMetadata(AmazonItemMetadata):
    config: ClassVar[AmazonBooksItemMetadataConfig] = AmazonBooksItemMetadataConfig()
