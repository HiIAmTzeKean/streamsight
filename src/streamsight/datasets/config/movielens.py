from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import DatasetConfig, MetadataConfig


@dataclass
class MovieLensDatasetConfig(DatasetConfig):
    """MovieLens base configuration."""

    user_ix: str = "userId"
    item_ix: str = "movieId"
    timestamp_ix: str = "timestamp"
    rating_ix: str = "rating"
    """Name of the column in the DataFrame that contains the rating a user gave to the item."""
    dataset_url: str = "https://files.grouplens.org/datasets/movielens"
    remote_zipname: str = "ml-100k"
    """Name of the zip-file on the MovieLens server."""
    remote_filename: str = "ratings.csv"
    """Name of the file containing user ratings on the MovieLens server."""
    default_base_path: str = DatasetConfig.default_base_path + "/movielens"


@dataclass
class MovieLens100KDatasetConfig(MovieLensDatasetConfig):
    """MovieLens 100K specific configuration."""

    remote_filename: str = "u.data"


@dataclass
class MovieLens100kUserMetadataConfig(MetadataConfig, MovieLensDatasetConfig):
    """
    MovieLens 100K User Metadata Configuration.

    Handles configuration for user demographic data:
    - User ID mapping
    - Age information
    - Gender information
    - Occupation information
    - Zipcode information

    All properties are computed from base fields to ensure consistency.
    """
    user_ix: str = "userId"
    """Name of the column containing user identifiers."""
    age_ix: str = "age"
    """Name of the column containing user age."""
    gender_ix: str = "gender"
    """Name of the column containing user gender."""
    occupation_ix: str = "occupation"
    """Name of the column containing user occupation."""
    zipcode_ix: str = "zipcode"
    """Name of the column containing user zipcode."""

    remote_filename: str = "u.user"
    """Filename of user metadata file in remote zip."""
    remote_zipname: str = "ml-100k"
    """Name of the zip file on remote server."""
    dataset_url: str = "https://files.grouplens.org/datasets/movielens"
    """URL to fetch the metadata from."""

    @property
    def column_names(self) -> list[str]:
        return [
            self.user_ix,
            self.age_ix,
            self.gender_ix,
            self.occupation_ix,
            self.zipcode_ix,
        ]

    @property
    def dtype_dict(self) -> dict:
        return {
            self.age_ix: np.int64,
            self.gender_ix: str,
            self.occupation_ix: str,
            self.zipcode_ix: str,
        }


@dataclass
class MovieLens100kItemMetadataConfig(MetadataConfig, MovieLensDatasetConfig):
    """
    MovieLens 100K Item Metadata Configuration.

    Handles configuration for movie metadata including:
    - Movie ID mapping
    - Title, release date, IMDB URL
    - 19 binary genre indicator columns

    All properties are computed from base fields to ensure consistency.
    """

    item_ix: str = "movieId"
    """Name of the column containing movie identifiers."""
    title_ix: str = "title"
    """Name of the column containing movie title."""
    release_date_ix: str = "releaseDate"
    """Name of the column containing movie release date."""
    video_release_date_ix: str = "videoReleaseDate"
    """Name of the column containing video release date."""
    imdb_url_ix: str = "imdbUrl"
    """Name of the column containing IMDB URL."""
    genres: tuple[str, ...] = (
        "unknown",
        "action",
        "adventure",
        "animation",
        "children",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "filmNoir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sciFi",
        "thriller",
        "war",
        "western",
    )
    """Tuple of 19 genre names in canonical order."""

    remote_filename: str = "u.item"
    remote_zipname: str = "ml-100k"
    dataset_url: str = "https://files.grouplens.org/datasets/movielens"
    encoding: str = "ISO-8859-1"
    """File encoding (ISO-8859-1 needed for special characters)."""

    @property
    def non_genre_columns(self) -> list[str]:
        """
        Column names for non-genre metadata.

        Returns:
            list[str]: [movie_id, title, release_date, video_release_date, imdb_url]

        Example:
            ["movieId", "title", "releaseDate", "videoReleaseDate", "imdbUrl"]
        """
        return [
            self.item_ix,
            self.title_ix,
            self.release_date_ix,
            self.video_release_date_ix,
            self.imdb_url_ix,
        ]

    @property
    def column_names(self) -> list[str]:
        return self.non_genre_columns + list(self.genres)

    @property
    def dtype_dict(self) -> dict:
        dtype_dict: dict[str, Any] = {
            self.title_ix: str,
            self.release_date_ix: str,
            self.video_release_date_ix: str,
            self.imdb_url_ix: str,
        }
        dtype_dict.update({genre: np.int64 for genre in self.genres})
        return dtype_dict
