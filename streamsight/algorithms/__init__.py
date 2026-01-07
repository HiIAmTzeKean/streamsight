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
    Popularity

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
    ItemKNNRolling
    ItemKNNStatic
"""
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.random import Random
from streamsight.algorithms.popularity import Popularity
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.algorithms.itemknn_incremental import ItemKNNIncremental
from streamsight.algorithms.itemknn_rolling import ItemKNNRolling
from streamsight.algorithms.itemknn_static import ItemKNNStatic