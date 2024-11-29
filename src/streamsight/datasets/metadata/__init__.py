"""Metadata module for dataset information.

This module allows users to include metadata information corresponding to datasets.
Metadata classes are built on top of the `Metadata` base class, allowing for easy
extension and customization.

## Important Notes

User and item IDs in the metadata module are mapped according to Streamsight's
internal mapping, not the original IDs. Developers should not load metadata from
source separately. Instead, implement the metadata class and load metadata while
loading the dataset.

## Available Metadata

- `Metadata`: Abstract base class for metadata implementations
- `MovieLens100kUserMetadata`: User metadata from MovieLens 100K dataset
- `MovieLens100kItemMetadata`: Item metadata from MovieLens 100K dataset
- `AmazonBookItemMetadata`: Item metadata from Amazon Books dataset
- `AmazonMovieItemMetadata`: Item metadata from Amazon Movies dataset
- `AmazonMusicItemMetadata`: Item metadata from Amazon Music dataset
- `LastFMUserMetadata`: User metadata from Last.FM dataset
- `LastFMItemMetadata`: Item metadata from Last.FM dataset
- `LastFMTagMetadata`: Tag metadata from Last.FM dataset

## Example

Load metadata from the MovieLens 100K dataset:

```python
from streamsight.datasets.movielens import MovieLens100K

dataset = MovieLens100K(fetch_dataset=True)
data = dataset.load()
```
"""

from .amazon import (
    AmazonBookItemMetadata,
    AmazonMovieItemMetadata,
    AmazonMusicItemMetadata,
)
from .base import Metadata
from .lastfm import LastFMItemMetadata, LastFMTagMetadata, LastFMUserMetadata
from .movielens import MovieLens100kItemMetadata, MovieLens100kUserMetadata


__all__ = [
    "Metadata",
    "MovieLens100kUserMetadata",
    "MovieLens100kItemMetadata",
    "AmazonBookItemMetadata",
    "AmazonMovieItemMetadata",
    "AmazonMusicItemMetadata",
    "LastFMUserMetadata",
    "LastFMItemMetadata",
    "LastFMTagMetadata",
]