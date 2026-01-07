import logging
import os
import zipfile
import pandas as pd
import numpy as np
from streamsight.datasets.base import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
tqdm.pandas()

class MovieLensDataset(Dataset):
    """Base class for Movielens datasets.
    
    Other Movielens datasets should inherit from this class.
    
    This code is adapted from RecPack :cite:`recpack`
    
    """
    USER_IX = "userId"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "movieId"
    """Name of the column in the DataFrame that contains item identifiers."""
    TIMESTAMP_IX = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    RATING_IX = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""

    DATASETURL = "http://files.grouplens.org/datasets/movielens"

    REMOTE_ZIPNAME = ""
    """Name of the zip-file on the MovieLens server."""

    REMOTE_FILENAME = "ratings.csv"
    """Name of the file containing user ratings on the MovieLens server."""
    
    @property
    def DEFAULT_FILENAME(self) -> str:
        """Default filename that will be used if it is not specified by the user."""
        return f"{self.REMOTE_ZIPNAME}_{self.REMOTE_FILENAME}"
        
    def _download_dataset(self):
        """Downloads the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        # Download the zip into the data directory
        self._fetch_remote(
            f"{self.DATASETURL}/{self.REMOTE_ZIPNAME}.zip", os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip")
        )

        # Extract the ratings file which we will use
        with zipfile.ZipFile(os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}.zip"), "r") as zip_ref:
            zip_ref.extract(f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}", self.base_path)

        # Rename the ratings file to the specified filename
        os.rename(os.path.join(self.base_path, f"{self.REMOTE_ZIPNAME}/{self.REMOTE_FILENAME}"), self.file_path)
class MovieLens100K(MovieLensDataset):

    REMOTE_FILENAME = "u.data"
    REMOTE_ZIPNAME = "ml-100k"

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        df = pd.read_table(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
                self.TIMESTAMP_IX: np.int64,
            },
            sep="\t",
            names=[self.USER_IX, self.ITEM_IX, self.RATING_IX, self.TIMESTAMP_IX],
        )

        return df