import os
import zipfile
from typing import ClassVar

import pandas as pd

from streamsight.datasets.config import (
    MovieLens100kItemMetadataConfig,
    MovieLens100kUserMetadataConfig,
)
from .base import Metadata


class MovieLens100kMetadata(Metadata):
    def _download_dataset(self) -> None:
        # Download the zip into the data directory
        self._fetch_remote(
            f"{self.config.dataset_url}/{self.config.remote_zipname}.zip",
            os.path.join(self.base_path, f"{self.config.remote_zipname}.zip"),
        )

        # Extract the ratings file which we will use
        with zipfile.ZipFile(
            os.path.join(self.base_path, f"{self.config.remote_zipname}.zip"), "r"
        ) as zip_ref:
            zip_ref.extract(
                f"{self.config.remote_zipname}/{self.config.remote_filename}", self.base_path
            )

        # Rename the ratings file to the specified filename
        os.rename(
            os.path.join(
                self.base_path, f"{self.config.remote_zipname}/{self.config.remote_filename}"
            ),
            self.file_path,
        )


class MovieLens100kUserMetadata(MovieLens100kMetadata):
    config: ClassVar[MovieLens100kUserMetadataConfig] = MovieLens100kUserMetadataConfig()  # type: ignore

    def __init__(self, user_id_mapping: pd.DataFrame) -> None:
        super().__init__()
        self.user_id_mapping = user_id_mapping

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        df = pd.read_table(
            self.file_path,
            dtype=self.config.dtype_dict,
            sep=self.config.sep,
            names=self.config.column_names,
            converters={self.config.user_ix: self._map_user_id},
        )
        return df

    def _map_user_id(self, user_id):
        user_id_to_uid = dict(
            zip(self.user_id_mapping[self.config.user_ix], self.user_id_mapping["uid"])
        )
        return user_id_to_uid.get(int(user_id), user_id)


class MovieLens100kItemMetadata(MovieLens100kMetadata):
    config: ClassVar[MovieLens100kItemMetadataConfig] = MovieLens100kItemMetadataConfig()  # type: ignore

    def __init__(self, item_id_mapping: pd.DataFrame) -> None:
        super().__init__()
        self.item_id_mapping = item_id_mapping

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        df = pd.read_table(
            self.file_path,
            dtype=self.config.dtype_dict,
            sep=self.config.sep,
            names=self.config.column_names,
            converters={self.config.item_ix: self._map_item_id},
            encoding=self.config.encoding,
        )
        return df

    def _map_item_id(self, item_id):
        item_id_to_iid = dict(
            zip(self.item_id_mapping[self.config.item_ix], self.item_id_mapping["iid"])
        )
        return item_id_to_iid.get(int(item_id), item_id)
