"""
.. currentmodule:: streamsight.datasets

Dataset
-------------

The dataset module allows users to easily use to publicly available datasets
in their experiments. The dataset class are built on top of the :class:`Dataset`
allowing for easy extension and customization. In this module, we provide
the a few dataset that is available from public sources. The programmer is free to add more
datasets as they see fit by defining the abstract methods that must be implemented.

Other than the publicly available datasets, we also provide a test dataset
that can be used for testing purposes. The test dataset is a simple dataset
that can be used to test the functionality of the algorithms.

While the MovieLens100K dataset is available in the module, we recommend that
the programmer use the other publicly available datasets as the data are not
chunked into "blocks". The setting of a global timeline to split the data
could potentially cause a chuck of data to be lost.

.. autosummary::
    :toctree: generated/

    Dataset
    TestDataset
    AmazonBookDataset
    AmazonComputerDataset
    AmazonMovieDataset
    AmazonMusicDataset
    YelpDataset
    MovieLens100K
    
Example
~~~~~~~~~

If the file specified does not exist, the dataset is downloaded and written into this file.
Subsequent loading of the dataset will not require downloading the dataset again,
and will be obtained from the file in the directory.

.. code-block:: python

    from streamsight.datasets import AmazonMusicDataset

    dataset = AmazonMusicDataset()
    data = dataset.load()


Each dataset can be loaded with default filters that are applied to the dataset.
To use the default filters, set `use_default_filters` parameter to True.
The dataset can also be loaded without filters and preprocessing of ID by
calling the :meth:`load` method with the parameter `apply_filters` set to False.
The recommended loading is with filters applied to ensure that the user and item
ids are incrementing in the order of time.

.. code-block:: python

    from streamsight.datasets import AmazonMusicDataset

    dataset = AmazonMusicDataset(use_default_filters=True)
    data = dataset.load(apply_filters=False)

For an overview of available filters see :mod:`streamsight.preprocessing`
"""

from streamsight.datasets.base import Dataset
from streamsight.datasets.test import TestDataset
from streamsight.datasets.amazon import (
    AmazonBookDataset,
    AmazonComputerDataset,
    AmazonMovieDataset,
    AmazonMusicDataset,
)
from streamsight.datasets.yelp import YelpDataset
from streamsight.datasets.movielens import MovieLens100K
