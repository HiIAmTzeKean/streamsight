import logging
import os
import zipfile
from typing import ClassVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import Dataset
from .config import MovieLens100KDatasetConfig, MovieLensDatasetConfig
from .metadata.movielens import MovieLens100kItemMetadata, MovieLens100kUserMetadata


logger = logging.getLogger(__name__)


class MovieLensDataset(Dataset):
    """Base class for Movielens datasets.

    Other Movielens datasets should inherit from this class.

    This code is adapted from RecPack :cite:`recpack`
    """
    config: ClassVar[MovieLensDatasetConfig] = MovieLensDatasetConfig()

    def _download_dataset(self) -> None:
        # Download the zip into the data directory
        zip_file_path = os.path.join(self.config.default_base_path, f"{self.config.remote_zipname}.zip")
        self._fetch_remote(
            url=f"{self.config.dataset_url}/{self.config.remote_zipname}.zip",
            filename=zip_file_path,
        )

        # Extract the ratings file which we will use
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extract(f"{self.config.remote_zipname}/{self.config.remote_filename}", self.config.default_base_path)

        # Rename the ratings file to the specified filename
        os.rename(
            os.path.join(self.config.default_base_path, f"{self.config.remote_zipname}/{self.config.remote_filename}"),
            self.file_path,
        )


class MovieLens100K(MovieLensDataset):
    """MovieLens 100K dataset."""
    ITEM_METADATA = None
    USER_METADATA = None

    config: ClassVar[MovieLens100KDatasetConfig] = MovieLens100KDatasetConfig()

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        chunks = pd.read_table(
            self.file_path,
            dtype={
                self.config.user_ix: np.int64,
                self.config.item_ix: np.int64,
                self.config.rating_ix: np.float64,
                self.config.timestamp_ix: np.int64,
            },
            sep="\t",
            names=[
                self.config.user_ix,
                self.config.item_ix,
                self.config.rating_ix,
                self.config.timestamp_ix,
            ],
            chunksize=100_000,
        )
        df = pd.concat(
            [chunk for chunk in tqdm(chunks, desc="Reading table", unit="chunk")], ignore_index=True
        )
        return df

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        self.USER_METADATA = MovieLens100kUserMetadata(user_id_mapping=user_id_mapping).load()
        self.ITEM_METADATA = MovieLens100kItemMetadata(item_id_mapping=item_id_mapping).load()
