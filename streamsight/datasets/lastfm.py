import logging
import os
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from streamsight.datasets.base import Dataset

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
    USER_IX = "userID"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "artistID"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    TAG_IX = "tagID"
    """Name of the column in the DataFrame that contains the tag a user gave to the item."""
    REMOTE_FILENAME = "user_taggedartists-timestamps.dat"
    """Name of the file containing user interaction on the MovieLens server."""
    REMOTE_ZIPNAME = "hetrec2011-lastfm-2k"
    """Name of the zip-file on the MovieLens server."""
    DATASET_URL = "https://files.grouplens.org/datasets/hetrec2011"
    """URL to fetch the dataset from."""

    @property
    def DEFAULT_FILENAME(self) -> str:
        """Default filename that will be used if it is not specified by the user."""
        return self.REMOTE_FILENAME

    def fetch_dataset(self, force=False) -> None:
        """Check if dataset is present, if not download

        :param force: If True, dataset will be downloaded,
                even if the file already exists.
                Defaults to False.
        :type force: bool, optional
        """
        path = os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip")
        file = self.REMOTE_FILENAME
        if not os.path.exists(path) or force:
            logger.debug(f"{self.name} dataset zipfile not found in {path}.")
            self._download_dataset()
        elif not os.path.exists(self.file_path) or force:
            logger.debug(
                f"{self.name} dataset file not found, but the zipfile has already been downloaded. Extracting file from zipfile."
            )
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extract(file, self.base_path)

        logger.debug(f"Data zipfile is in memory and in dir specified.")

    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the interaction file to `self.file_path`
        """
        # Download the zip into the data directory
        self._fetch_remote(
            f"{self.DATASET_URL}/{self.REMOTE_ZIPNAME}.zip",
            os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"),
        )

        # Extract the interaction file which we will use
        with zipfile.ZipFile(
            os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r"
        ) as zip_ref:
            zip_ref.extract(f"{self.REMOTE_FILENAME}", self.base_path)

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
                self.ITEM_IX: np.int32,
                self.USER_IX: np.int32,
                self.TAG_IX: np.int32,
                self.TIMESTAMP_IX: np.int64,
            },
            sep="\t",
            names=[
                self.USER_IX,
                self.ITEM_IX,
                self.TAG_IX,
                self.TIMESTAMP_IX,
            ],
            header=0,
        )
        return df
