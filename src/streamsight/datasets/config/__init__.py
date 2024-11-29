"""Dataset configuration module.

This module provides configuration classes for dataset loading and metadata
handling. Configurations define dataset properties such as paths, URLs, and
processing parameters.

## Available Configurations

### Base Classes

- `DatasetConfig`: Base class for dataset configurations
- `MetadataConfig`: Base class for metadata configurations

### Dataset Configurations

- `MovieLensDatasetConfig`: Base configuration for MovieLens datasets
- `MovieLens100KDatasetConfig`: Configuration for MovieLens 100K dataset
- `AmazonDatasetConfig`: Base configuration for Amazon datasets
- `AmazonMusicDatasetConfig`: Configuration for Amazon Music dataset
- `AmazonMovieDatasetConfig`: Configuration for Amazon Movies dataset
- `AmazonBookDatasetConfig`: Configuration for Amazon Books dataset
- `AmazonSubscriptionBoxesDatasetConfig`: Configuration for Amazon Subscription Boxes dataset
- `LastFMDatasetConfig`: Configuration for Last.FM dataset
- `YelpDatasetConfig`: Configuration for Yelp dataset

### Metadata Configurations

- `MovieLens100kItemMetadataConfig`: Item metadata configuration for MovieLens 100K
- `MovieLens100kUserMetadataConfig`: User metadata configuration for MovieLens 100K
- `AmazonItemMetadataConfig`: Base Amazon item metadata configuration
- `AmazonBooksItemMetadataConfig`: Item metadata for Amazon Books
- `AmazonDigitalMusicItemMetadataConfig`: Item metadata for Amazon Digital Music
- `AmazonMoviesAndTVItemMetadataConfig`: Item metadata for Amazon Movies and TV
- `AmazonSubscriptionBoxesItemMetadataConfig`: Item metadata for Amazon Subscription Boxes
- `LastFMUserMetadataConfig`: User metadata configuration for Last.FM
- `LastFMItemMetadataConfig`: Item metadata configuration for Last.FM
- `LastFMTagMetadataConfig`: Tag metadata configuration for Last.FM

## Usage

A typical usage pattern is to import a dataset config, optionally override fields,
and pass it to dataset-loading utilities or custom convenience wrappers:

```python
from streamsight.datasets.config import AmazonMusicDatasetConfig

# Create config instance using defaults
cfg = AmazonMusicDatasetConfig()

# Inspect config values
print(cfg.name)
print(cfg.local_path)
print(cfg.source_url)

# Optionally override defaults at runtime
custom_cfg = AmazonMusicDatasetConfig(
    min_user_interactions=5,
    min_item_interactions=10
)
```
"""

from .amazon import (
    AmazonBookDatasetConfig,
    AmazonBooksItemMetadataConfig,
    AmazonDatasetConfig,
    AmazonDigitalMusicItemMetadataConfig,
    AmazonItemMetadataConfig,
    AmazonMovieDatasetConfig,
    AmazonMoviesAndTVItemMetadataConfig,
    AmazonMusicDatasetConfig,
    AmazonSubscriptionBoxesDatasetConfig,
    AmazonSubscriptionBoxesItemMetadataConfig,
)
from .base import DatasetConfig, MetadataConfig
from .lastfm import (
    LastFMDatasetConfig,
    LastFMItemMetadataConfig,
    LastFMTagMetadataConfig,
    LastFMUserMetadataConfig,
)
from .movielens import (
    MovieLens100KDatasetConfig,
    MovieLens100kItemMetadataConfig,
    MovieLens100kUserMetadataConfig,
    MovieLensDatasetConfig,
)
from .yelp import YelpDatasetConfig


__all__ = [
    "AmazonDatasetConfig",
    "AmazonMusicDatasetConfig",
    "AmazonMovieDatasetConfig",
    "AmazonBookDatasetConfig",
    "AmazonSubscriptionBoxesDatasetConfig",
    "LastFMDatasetConfig",
    "YelpDatasetConfig",
    "DatasetConfig",
    "MetadataConfig",
    "MovieLensDatasetConfig",
    "MovieLens100KDatasetConfig",
    "MovieLens100kItemMetadataConfig",
    "MovieLens100kUserMetadataConfig",
    "AmazonBooksItemMetadataConfig",
    "AmazonDigitalMusicItemMetadataConfig",
    "AmazonItemMetadataConfig",
    "AmazonMoviesAndTVItemMetadataConfig",
    "AmazonSubscriptionBoxesItemMetadataConfig",
    "LastFMItemMetadataConfig",
    "LastFMTagMetadataConfig",
    "LastFMUserMetadataConfig",
]
