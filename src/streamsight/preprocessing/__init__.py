"""Preprocessing module for data preparation.

This module contains filters and preprocessors for preparing data before
transforming it into an InteractionMatrix object.

## Filters

Filters are used to filter data before transforming it into an InteractionMatrix
object. Filter implementations must extend the abstract `Filter` class.

Available filters:

- `Filter`: Abstract base class for filter implementations
- `MinItemsPerUser`: Filter requiring minimum interactions per user
- `MinUsersPerItem`: Filter requiring minimum interactions per item

## Preprocessor

The preprocessor allows adding filters for data preprocessing and manages ID
mappings. After applying filters, it updates item and user ID mappings to
internal IDs to reduce computation load and enable easy matrix representation.

Available preprocessor:

- `DataFramePreprocessor`: Preprocesses pandas DataFrames into InteractionMatrix
"""

from streamsight.preprocessing.filter import Filter, MinItemsPerUser, MinUsersPerItem
from streamsight.preprocessing.preprocessor import DataFramePreprocessor


__all__ = [
    "Filter",
    "MinItemsPerUser",
    "MinUsersPerItem",
    "DataFramePreprocessor",
]
