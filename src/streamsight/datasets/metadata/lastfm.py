import logging
import os
import zipfile
from typing import ClassVar

import pandas as pd

from streamsight.datasets.config import (
    LastFMItemMetadataConfig,
    LastFMTagMetadataConfig,
    LastFMUserMetadataConfig,
)
from .base import Metadata


logger = logging.getLogger(__name__)


class LastFMMetadata(Metadata):
    config: ClassVar[LastFMUserMetadataConfig] = LastFMUserMetadataConfig()  # type: ignore

    def _download_dataset(self) -> None:
        """Downloads the metadata for the dataset.

        Downloads the zipfile, and extracts the ratings file to `self.file_path`
        """
        # Download the zip into the data directory
        self._fetch_remote(
            f"{self.config.dataset_url}/{self.config.remote_filename}.zip",
            os.path.join(self.base_path, f"{self.config.remote_filename}.zip"),
        )

        # Extract the interaction file which we will use
        with zipfile.ZipFile(
            os.path.join(self.base_path, f"{self.config.remote_filename}.zip"), "r"
        ) as zip_ref:
            zip_ref.extract(f"{self.config.remote_filename}", self.base_path)


class LastFMUserMetadata(Metadata):
    config: ClassVar[LastFMUserMetadataConfig] = LastFMUserMetadataConfig()  # type: ignore

    def __init__(self, user_id_mapping: pd.DataFrame) -> None:
        super().__init__()
        self.user_id_mapping = user_id_mapping

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        df = pd.read_csv(
            self.file_path,
            sep=self.config.sep,
            names=self.config.column_names,
            converters={
                self.config.user_ix: self._map_user_id,
                self.config.friend_ix: self._map_user_id,
            },
            header=0,
        )
        return df

    def _map_user_id(self, user_id):
        user_id_to_uid = dict(
            zip(self.user_id_mapping[self.config.user_ix], self.user_id_mapping["uid"])
        )
        return user_id_to_uid.get(int(user_id), user_id)


class LastFMItemMetadata(Metadata):
    config: ClassVar[LastFMItemMetadataConfig] = LastFMItemMetadataConfig()  # type: ignore

    def __init__(self, item_id_mapping: pd.DataFrame) -> None:
        super().__init__()
        self.item_id_mapping = item_id_mapping

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        df = pd.read_csv(
            self.file_path,
            dtype=self.config.dtype_dict,
            sep=self.config.sep,
            names=self.config.column_names,
            converters={
                self.config.item_ix: self._map_item_id,
            },
            header=0,
        )
        return df

    def _map_item_id(self, item_id):
        item_id_to_iid = dict(zip(self.item_id_mapping["artistID"], self.item_id_mapping["iid"]))
        return item_id_to_iid.get(int(item_id), item_id)


class LastFMTagMetadata(Metadata):
    config: ClassVar[LastFMTagMetadataConfig] = LastFMTagMetadataConfig()  # type: ignore

    def __init__(self) -> None:
        super().__init__()

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        df = pd.read_csv(
            self.file_path,
            dtype=self.config.dtype_dict,
            sep=self.config.sep,
            names=self.config.column_names,
            encoding="ISO-8859-1",
            header=0,
        )
        return df
