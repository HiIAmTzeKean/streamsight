import logging
import os
import pandas as pd
import numpy as np
from streamsight.datasets.base import Dataset

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
    USER_IX = "user_id"
    """Name of the column in the DataFrame that contains user identifiers."""
    ITEM_IX = "business_id"
    """Name of the column in the DataFrame that contains item identifiers."""
    RATING_IX = "stars"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    TIMESTAMP_IX = "date"
    """Name of the column in the DataFrame that contains time of interaction in date format."""
    
    DEFAULT_FILENAME = "yelp_academic_dataset_review.csv"
    """Default filename that will be used if it is not specified by the user."""

    DATASET_URL = "https://www.yelp.com/dataset/download"
    """URL to fetch the dataset from."""
    
    
    def _download_dataset(self):
        raise ValueError(f"Yelp dataset has not been downloaded. Please head over"
                         f"to {self.DATASET_URL} to download the dataset."
                         "As there is a license agreement, we cannot download it for you."
                         "Place the unzip dataset under the data directory when done.")

    
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
                self.ITEM_IX: str,
                self.USER_IX: str,
                self.RATING_IX: np.float32,
                self.TIMESTAMP_IX: str
            },
            usecols=[self.ITEM_IX, self.USER_IX, self.RATING_IX, self.TIMESTAMP_IX],
            parse_dates=[self.TIMESTAMP_IX],
            date_format='%Y-%m-%d %H:%M:%S',
            header=0,
            sep=",",
            encoding='utf-8'
        )
        
        # remove the byte literal char from the string columns
        str_df = df.select_dtypes(["object"])
        str_df = str_df.stack().str[2:-1].unstack()
        for col in str_df:
            df[col] = str_df[col]

        # convert the timestamp to epoch time
        df[self.TIMESTAMP_IX] = pd.to_datetime(df[self.TIMESTAMP_IX], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df.dropna(inplace=True)
        df[self.TIMESTAMP_IX] = df[self.TIMESTAMP_IX].astype(np.int64) // 10**9
        
        return df
