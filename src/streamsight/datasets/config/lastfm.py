from dataclasses import dataclass

from .base import DatasetConfig, MetadataConfig


@dataclass
class LastFMDatasetConfig(DatasetConfig):
    """LastFM dataset configuration."""

    user_ix: str = "userID"
    item_ix: str = "artistID"
    timestamp_ix: str = "timestamp"
    tag_ix: str = "tagID"
    """Name of the column in the DataFrame that contains the tag a user gave to the item."""
    dataset_url: str = "https://files.grouplens.org/datasets/hetrec2011"
    remote_zipname: str = "hetrec2011-lastfm-2k"
    remote_filename: str = "user_taggedartists-timestamps.dat"
    default_base_path: str = DatasetConfig.default_base_path + "/lastfm"


@dataclass
class LastFMUserMetadataConfig(MetadataConfig, LastFMDatasetConfig):
    """LastFM User Metadata Configuration."""

    user_ix: str = "userID"
    """Name of the column containing user identifiers."""
    friend_ix: str = "friendID"
    """Name of the column containing friend identifiers."""

    remote_filename: str = "user_friends.dat"
    remote_zipname: str = "hetrec2011-lastfm-2k"
    dataset_url: str = "https://files.grouplens.org/datasets/hetrec2011"
    sep: str = "\t"

    @property
    def column_names(self) -> list[str]:
        return [
            self.user_ix,
            self.friend_ix,
        ]


@dataclass
class LastFMItemMetadataConfig(MetadataConfig, LastFMDatasetConfig):
    """LastFM Item Metadata Configuration."""

    item_ix: str = "id"
    """Name of the column containing item identifiers."""
    name_ix: str = "name"
    """Name of the column containing item names."""
    url_ix: str = "url"
    """Name of the column containing item URLs."""
    picture_url_ix: str = "pictureURL"
    """Name of the column containing item picture URLs."""

    remote_filename: str = "artists.dat"
    remote_zipname: str = "hetrec2011-lastfm-2k"
    dataset_url: str = "https://files.grouplens.org/datasets/hetrec2011"
    sep: str = "\t"

    @property
    def column_names(self) -> list[str]:
        return [
            self.item_ix,
            self.name_ix,
            self.url_ix,
            self.picture_url_ix,
        ]

    @property
    def dtype_dict(self) -> dict:
        return {
            self.name_ix: str,
            self.url_ix: str,
            self.picture_url_ix: str,
        }


@dataclass
class LastFMTagMetadataConfig(MetadataConfig, LastFMDatasetConfig):
    """LastFM Tag Metadata Configuration."""

    tag_ix: str = "tagID"
    """Name of the column containing tag identifiers."""
    name_ix: str = "tagValue"
    """Name of the column containing tag names."""

    remote_filename: str = "tags.dat"
    remote_zipname: str = "hetrec2011-lastfm-2k"
    dataset_url: str = "https://files.grouplens.org/datasets/hetrec2011"
    sep: str = "\t"

    @property
    def column_names(self) -> list[str]:
        return [
            self.tag_ix,
            self.name_ix,
        ]

    @property
    def dtype_dict(self) -> dict:
        return {
            self.tag_ix: str,
            self.name_ix: str,
        }
