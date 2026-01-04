"""
streamsight.matrix

This module provides classes and utilities for handling interaction matrices in recommendation systems.

The core component is the InteractionMatrix, which represents user-item interactions as a structured matrix.
It stores interaction data in a pandas DataFrame and offers methods for filtering, masking, and converting to sparse matrices.
This is essential for building and evaluating recommender algorithms, such as collaborative filtering, where interactions
between users and items need to be efficiently processed.

Use cases include:
- Preprocessing interaction data for training recommendation models.
- Handling temporal data with timestamp-based filtering (e.g., recent interactions).
- Masking unknown users/items during evaluation to prevent data leakage.
- Converting data to CSR format for efficient matrix operations in libraries like SciPy.

Classes:
    InteractionMatrix: The main class for creating and manipulating interaction matrices from datasets.
        It supports operations like filtering by users/items, timestamps, and shape masking.
    PredictionMatrix: A subclass of InteractionMatrix tailored for prediction-related operations.
        It provides masking for the expected (user, item) exposed.

Enums:
    ItemUserBasedEnum: Enum for specifying whether operations are item-based or user-based.
        Used in methods that group or filter data by users or items.

Exceptions:
    TimestampAttributeMissingError: Raised when required timestamp attributes are missing from the data.
        Ensures that time-aware operations are only performed on timestamped data.

Functions:
    to_csr_matrix: Utility function to convert data structures to CSR matrix format.
        Useful for creating sparse representations of interaction data for computational efficiency.
"""

from .exception import TimestampAttributeMissingError
from .interaction_matrix import InteractionMatrix, ItemUserBasedEnum
from .prediction_matrix import PredictionMatrix
from .util import to_csr_matrix


__all__ = [
    "InteractionMatrix",
    "PredictionMatrix",
    "to_csr_matrix",
    "ItemUserBasedEnum",
    "TimestampAttributeMissingError",
]
