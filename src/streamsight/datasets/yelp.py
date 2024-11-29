import logging
from typing import ClassVar, NoReturn

import numpy as np
import pandas as pd

from streamsight.datasets.base import Dataset
from streamsight.datasets.config.yelp import YelpDatasetConfig


logger = logging.getLogger(__name__)


class YelpDataset(Dataset):
    """Yelp dataset

    The Yelp dataset contains user reviews of businesses. The main columns that
    will be used are:

    - user_id: The user identifier
    - business_id: The business identifier
    - stars: The rating given by the user to the business
    - date: The date of the review

    The dataset can be downloaded from https://www.yelp.com/dataset/download.
    The dataset is in a zip file, there are online codes that will aid you in
    converting the json file to a csv file for usage. Note that for the purposes
    of this class, it is assumed that the dataset has been converted to a csv file
    and is named `yelp_academic_dataset_review.csv`.

    Reference is made to the following code from the official repo from
    Yelp: https://github.com/Yelp/dataset-examples/blob/master/json_to_csv_converter.py

    you can use the following command to convert the json file to a csv file:

    .. code-block:: shell
        python json_to_csv_converter.py yelp_academic_dataset_review.json

    """

    config: ClassVar[YelpDatasetConfig] = YelpDatasetConfig()

    def _download_dataset(self) -> NoReturn:
        raise ValueError(
            "Yelp dataset has not been downloaded. Please head over"
            f"to {self.config.dataset_url} to download the dataset."
            "As there is a license agreement, we cannot download it for you."
            "Place the unzip dataset under the data directory when done."
            f"Expected filename: {self.config.default_filename}"
        )

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
                self.config.item_ix: str,
                self.config.user_ix: str,
                self.config.rating_ix: np.float32,
                self.config.timestamp_ix: str,
            },
            usecols=[
                self.config.item_ix,
                self.config.user_ix,
                self.config.rating_ix,
                self.config.timestamp_ix,
            ],
            parse_dates=[self.config.timestamp_ix],
            date_format="%Y-%m-%d %H:%M:%S",
            header=0,
            sep=",",
            encoding="utf-8",
        )

        # remove the byte literal char from the string columns
        str_df = df.select_dtypes(["object"])
        str_df = str_df.stack().str[2:-1].unstack()
        for col in str_df:
            df[col] = str_df[col]

        # convert the timestamp to epoch time
        df[self.config.timestamp_ix] = pd.to_datetime(
            df[self.config.timestamp_ix], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df.dropna(inplace=True)
        df[self.config.timestamp_ix] = df[self.config.timestamp_ix].astype(np.int64) // 10**9

        return df

    def _fetch_dataset_metadata(
        self, user_id_mapping: pd.DataFrame, item_id_mapping: pd.DataFrame
    ) -> None:
        pass
