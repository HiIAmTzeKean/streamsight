"""Metrics module for evaluating recommender system performance.

This module provides a collection of metrics for evaluating the performance of
recommender systems in streaming environments. Metrics are implemented as classes
that inherit from the `Metric` base class, allowing for easy extension and customization.

## Available Metrics

The following metrics are currently available:

- `PrecisionK`: Precision at K
- `RecallK`: Recall at K
- `DCGK`: Discounted Cumulative Gain at K
- `NDCGK`: Normalized Discounted Cumulative Gain at K
- `HitK`: Hit Rate at K

## Using Metrics

To use a metric, simply instantiate the corresponding class and call the `evaluate` method
with the predicted and ground truth rankings:

```python
from streamsight.metrics import PrecisionK

metric = PrecisionK(k=10)
score = metric.evaluate(
    predicted_ranking, ground_truth_ranking
)
```

The `evaluate` method returns a single float value representing the metric score.

## Extending the Framework

To add custom metrics, inherit from the `Metric` base class and implement the `evaluate` method.
Refer to the base class documentation for implementation details.

# Related Modules

- streamsight.evaluators: Evaluator classes for running metrics over data streams
"""

from .binary import HitK
from .core import ListwiseMetricK, Metric, MetricTopK
from .ranking import DCGK, NDCGK, PrecisionK, RecallK


__all__ = [
    "Metric",
    "PrecisionK",
    "RecallK",
    "DCGK",
    "NDCGK",
    "HitK",
    "ListwiseMetricK",
    "MetricTopK",
]
