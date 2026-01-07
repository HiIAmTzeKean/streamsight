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
"""

from streamsight.metrics.base import Metric
from streamsight.metrics.precision import PrecisionK
from streamsight.metrics.recall import RecallK