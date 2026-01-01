"""Dataset module for public datasets in streaming experiments.

This module provides easy access to publicly available datasets for use in streaming
experiments. Dataset classes are built on top of the `Dataset` base class, allowing
for easy extension and customization.

## Dataset Overview

Multiple public datasets are available from various sources. Additionally, a
lightweight test dataset is provided for testing algorithm functionality.

### Data Chunking Note

The MovieLens 100K dataset is available but not chunked into "blocks". Setting
a global timeline to split the data could potentially cause a chunk of data to
be lost. Other publicly available datasets are recommended.

## Available Datasets

- `AmazonBookDataset`: Amazon Books reviews
- `AmazonMovieDataset`: Amazon Movies reviews
- `AmazonMusicDataset`: Amazon Music reviews
- `AmazonSubscriptionBoxesDataset`: Amazon Subscription Boxes reviews
- `LastFMDataset`: Last.FM music listening history
- `MovieLens100K`: MovieLens 100K rating dataset
- `YelpDataset`: Yelp business reviews
- `TestDataset`: Lightweight dataset for testing algorithms

## Loading Datasets

Basic loading:

```python
from streamsight.datasets import AmazonMusicDataset

dataset = AmazonMusicDataset()
data = dataset.load()
```

If the file does not exist, it will be downloaded and written. Subsequent loads
will retrieve the file from disk without downloading again.

### Using Default Filters

```python
from streamsight.datasets import AmazonMusicDataset

dataset = AmazonMusicDataset(use_default_filters=True)
data = dataset.load(apply_filters=False)
```

Each dataset can be loaded with default filters applied. Default filters ensure
that user and item IDs increment in the order of time. **This is the recommended
loading approach.**

## Extending the Framework

To add custom datasets, inherit from `Dataset` and implement all abstract methods.
Refer to the base class documentation for implementation details.

## Related Modules

- `streamsight.preprocessing`: Data preprocessing and filtering utilities
"""

from .datasets import (
    AmazonBookDataset,
    AmazonMovieDataset,
    AmazonMusicDataset,
    AmazonSubscriptionBoxesDataset,
    Dataset,
    LastFMDataset,
    MovieLens100K,
    TestDataset,
    YelpDataset,
)
from .metadata import (
    AmazonBookItemMetadata,
    AmazonMovieItemMetadata,
    AmazonMusicItemMetadata,
    LastFMItemMetadata,
    LastFMTagMetadata,
    LastFMUserMetadata,
    Metadata,
    MovieLens100kItemMetadata,
    MovieLens100kUserMetadata,
)


__all__ = [
    "AmazonBookDataset",
    "AmazonMovieDataset",
    "AmazonMusicDataset",
    "AmazonSubscriptionBoxesDataset",
    "LastFMDataset",
    "MovieLens100K",
    "YelpDataset",
    "TestDataset",
    "Dataset",
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
