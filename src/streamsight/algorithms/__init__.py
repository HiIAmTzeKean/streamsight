"""
.. currentmodule:: streamsight.algorithms

The algorithms module in streamsight contains a collection of baseline algorithms
and various of the item-based KNN collaborative filtering algorithm. A total of
3 variation of the item-based KNN algorithm is implemented in the module. Which
are listed below

Algorithm
---------
Base class for all algorithms. Programmer should inherit from this class when
implementing a new algorithm. It provides a common interface for all algorithms
such that the expected methods and properties are defined to avoid any runtime
errors.

.. autosummary::
    :toctree: generated/

    Algorithm

Baseline Algorithms
-------------------

The baseline algorithms are simple algorithms that can be used as a reference
point to compare the performance of the more complex algorithms. The following
baseline algorithms are implemented in the module.

.. autosummary::
    :toctree: generated/

    Random
    RecentPopularity
    DecayPopularity
    MostPopular

Item Similarity Algorithms
----------------------------

Item similarity algorithms exploit relationships between items to make recommendations.
At prediction time, the user is represented by the items they have interacted
with. 3 variations of the item-based KNN algorithm are implemented in the module.
Each variation is to showcase the difference in the learning and prediction of
the algorithm. We note that no one algorithm is better than the other, and it
greatly depends on the dataset and parameters used in the algorithm which would
yield the best performance.

.. autosummary::
    :toctree: generated/

    ItemKNN
    ItemKNNIncremental
    ItemKNNIncrementalMovieLens100K
    ItemKNNRolling
    ItemKNNStatic

Time-Aware Item Similarity Algorithms
-------------------------------------

This is an extension of the item-based KNN algorithm.
The time-aware item-based KNN algorithms applies a decay factor to the timestamps of interactions.
This gives heavier weight to interactions that are more recent.

.. autosummary::
    :toctree: generated/

    TARSItemKNN
    TARSItemKNNLiu
    TARSItemKNNVaz
    TARSItemKNNDing
"""

from .base import Algorithm
from .baseline import MostPopular, Random, RecentPopularity
from .baseline.decay_popularity import DecayPopularity
from .itemknn import ItemKNN, ItemKNNIncremental, ItemKNNIncrementalMovieLens100K, ItemKNNRolling, ItemKNNStatic
from .time_aware_item_knn import (
    TARSItemKNN,
    TARSItemKNNDing,
    TARSItemKNNLiu,
    TARSItemKNNVaz,
)


__all__ = [
    "Algorithm",
    "DecayPopularity",
    "ItemKNN",
    "ItemKNNIncremental",
    "ItemKNNIncrementalMovieLens100K",
    "ItemKNNRolling",
    "ItemKNNStatic",
    "MostPopular",
    "Random",
    "RecentPopularity",
    "TARSItemKNN",
    "TARSItemKNNDing",
    "TARSItemKNNLiu",
    "TARSItemKNNVaz",
]
