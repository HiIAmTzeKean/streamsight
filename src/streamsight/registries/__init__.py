"""Registries for algorithms, metrics, and datasets.

This module provides registries for storing and managing algorithms, metrics,
and datasets used in experiments. Registries help keep track of valid classes
and enable easy instantiation of components.

## Registries

Registries store algorithms, metrics, and datasets by default and allow
registration of new components via the `register` function.

Example:
    ```python
    from streamsight.pipelines import ALGORITHM_REGISTRY
    from streamsight.algorithms import ItemKNNStatic

    algo = ALGORITHM_REGISTRY.get("ItemKNNStatic")(K=10)
    ALGORITHM_REGISTRY.register("algo_1", ItemKNNStatic)
    ```

### Available Registries

- `ALGORITHM_REGISTRY`: Registry for algorithms
- `DATASET_REGISTRY`: Registry for datasets
- `METRIC_REGISTRY`: Registry for metrics
- `AlgorithmRegistry`: Class for creating algorithm registries
- `DatasetRegistry`: Class for creating dataset registries
- `MetricRegistry`: Class for creating metric registries

## Entries

Entries store algorithms and metrics in registries. They maintain the class
and parameters used to instantiate each component. These entries are used by
`EvaluatorPipeline` to instantiate algorithms and metrics.

### Available Entries

- `AlgorithmEntry`: Entry for algorithms
- `MetricEntry`: Entry for metrics

"""

from .algorithm import (
    ALGORITHM_REGISTRY,
    AlgorithmEntry,
    AlgorithmRegistry,
)
from .base import Registry
from .dataset import DATASET_REGISTRY, DatasetRegistry
from .metric import (
    METRIC_REGISTRY,
    MetricEntry,
    MetricRegistry,
)


__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmEntry",
    "AlgorithmRegistry",
    "DATASET_REGISTRY",
    "DatasetRegistry",
    "METRIC_REGISTRY",
    "MetricEntry",
    "MetricRegistry",
    "Registry",
]
