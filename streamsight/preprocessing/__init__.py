"""
.. currentmodule:: streamsight.preprocessing

Filters
-------
The filter module contains classes that are used to filter the data before
transforming the data into an InteractionMatrix object.

.. autosummary::
    :toctree: generated/

    Filter
    MinItemsPerUser
    MinUsersPerItem

Preprocessor
------------
The preprocessor class allows the programmer to add the filters for data
preprocessing before transforming the data into an InteractionMatrix object.
The preprocessor class after applying the filters, updates the item and user
ID mappings into internal ID to reduce the computation load and allows for
easy representation of the matrix.

.. autosummary::
    :toctree: generated/

    DataFramePreprocessor
"""
from streamsight.preprocessing.preprocessor import DataFramePreprocessor
from streamsight.preprocessing.filter import Filter, MinItemsPerUser, MinUsersPerItem