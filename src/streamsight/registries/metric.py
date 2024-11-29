from typing import NamedTuple

import streamsight.metrics
from .base import Registry


class MetricRegistry(Registry):
    """Registry for easy retrieval of metric types by name.

    The registry comes preregistered with all the streamsight metrics.
    """

    def __init__(self) -> None:
        super().__init__(streamsight.metrics)


METRIC_REGISTRY = MetricRegistry()
"""Registry for metrics.

Contains the streamsight metrics by default and allows registration of new
metrics via the ``register`` function.

Example:
    ```python
    from streamsight.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get("Recall")(K=20)

    from streamsight.algorithms import Recall

    METRIC_REGISTRY.register("HelloWorld", Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get("HelloWorld")(K=20)
    ```
"""


class MetricEntry(NamedTuple):
    """Entry for the metric registry.

    The intended use of this class is to store the name of the metric and the
    top-K value for the metric specified by the user.

    Mainly this will happen during the building phase of the evaluator
    pipeline in :class:`Builder`.

    Attributes:
        name: Name of the algorithm.
        K: Top-K value for the metric.
    """

    name: str
    K: None | int = None
