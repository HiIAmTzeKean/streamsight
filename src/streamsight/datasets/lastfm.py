import logging
import os
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import ClassVar

from streamsight.datasets.base import Dataset
from streamsight.datasets.config import LastFMDatasetConfig
from .metadata.lastfm import LastFMItemMetadata, LastFMTagMetadata, LastFMUserMetadata


logger = logging.getLogger(__name__)
tqdm.pandas()


class LastFMDataset(Dataset):
    """
    Last FM dataset.

    The Last FM dataset contains user interactions with artists. The tags in this
    datasets are not used in this implementation. The dataset that will be used
    would the the user_taggedartists-timestamps.dat file. The dataset contains
    the following columns: [user, artist, tags, timestamp].

    The dataset is downloaded from the GroupLens website :cite:`Cantador_RecSys2011`.
    """

    config: ClassVar[LastFMDatasetConfig] = LastFMDatasetConfig()

    ITEM_METADATA = None
    USER_METADATA = None
    TAG_METADATA = None

    def fetch_dataset(self) -> None:
        """Check if dataset is present, if not download.

        This method overrides the base class to handle the special case where
        the zipfile may exist but the extracted file doesn't.
        """
        zip_path = os.path.join(
            self.config.default_base_path, f"{self.config.remote_zipname}.zip"
        )

        if not os.path.exists(zip_path):
            logger.debug(f"{self.name} dataset zipfile not found in {zip_path}.")
            self._download_dataset()
        elif not os.path.exists(self.file_path):
            logger.debug(
                f"{self.name} dataset file not found, but zipfile already downloaded. "
                f"Extracting file from zipfile."
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract(self.config.remote_filename, self.config.default_base_path)
        else:
            logger.debug("Data zipfile is in memory and in dir specified.")

    def _download_dataset(self) -> None:
        """Downloads the dataset.

        Downloads the zipfile, and extracts the interaction file to `self.file_path`
        """
        zip_path = os.path.join(
            self.config.default_base_path, f"{self.config.remote_zipname}.zip"
        )

        logger.debug(f"Downloading {self.name} dataset from {self.config.dataset_url}")

        # Download the zip into the data directory
        self._fetch_remote(
            f"{self.config.dataset_url}/{self.config.remote_zipname}.zip",
            zip_path,
        )

        # Extract the interaction file which we will use
        logger.debug(f"Extracting {self.config.remote_filename} from zip")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract(self.config.remote_filename, self.config.default_base_path)

    def _load_dataframe(self) -> pd.DataFrame:
        """Load the raw dataset from file, and return it as a pandas DataFrame.

        Transform the dataset downloaded to have integer user and item ids. This
        will be needed for representation in the interaction matrix.

        :return: The interaction data as a DataFrame with a row per interaction.
        :rtype: pd.DataFrame
        """
        self.fetch_dataset()
        df = pd.read_csv(
            self.file_path,
            dtype={
                self.config.item_ix: np.int32,
                self.config.user_ix: np.int32,
                self.config.tag_ix: np.int32,
                self.config.timestamp_ix: np.int64,
            },
            sep="\t",
            names=[
                self.config.user_ix,
                self.config.item_ix,
                self.config.tag_ix,
                self.config.timestamp_ix,
            ],
            header=0,
        )
        # Convert from milliseconds to seconds
        df[self.config.timestamp_ix] = df[self.config.timestamp_ix] // 1_000

        logger.debug(f"Loaded {len(df)} interactions")
        return df

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        self.USER_METADATA = LastFMUserMetadata(user_id_mapping=user_id_mapping).load()
        self.ITEM_METADATA = LastFMItemMetadata(item_id_mapping=item_id_mapping).load()
        self.TAG_METADATA = LastFMTagMetadata().load()
