"""
.. currentmodule:: streamsight.metrics

Metrics
----------

This module provides the functionality to evaluate the performance of the recommendation system.

.. autosummary::
    :toctree: generated/

    Metric
    PrecisionK
    RecallK
    DCGK
    NDCGK
    HitK
"""

from streamsight.metrics.base import Metric
from streamsight.metrics.dcg import DCGK
from streamsight.metrics.hit import HitK
from streamsight.metrics.ndcg import NDCGK
from streamsight.metrics.precision import PrecisionK
from streamsight.metrics.recall import RecallK
