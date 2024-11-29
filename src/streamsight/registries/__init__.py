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

## Status Registry

Status registry maintains algorithm status during the streaming process. It
tracks the state of algorithms and keeps state counts for monitoring.

### Status Components

- `AlgorithmStateEnum`: Enum for algorithm states
- `AlgorithmStatusEntry`: Entry for algorithm status
- `AlgorithmStatusRegistry`: Registry for algorithm status
"""

from .algorithm import (
    ALGORITHM_REGISTRY,
    AlgorithmEntry,
    AlgorithmRegistry,
    AlgorithmStateEnum,
    AlgorithmStatusEntry,
    AlgorithmStatusRegistry,
)
from .base import Registry
from .dataset import DATASET_REGISTRY, DatasetRegistry
from .metric import METRIC_REGISTRY, MetricEntry, MetricRegistry


__all__ = [
    "ALGORITHM_REGISTRY",
    "DATASET_REGISTRY",
    "METRIC_REGISTRY",
    "AlgorithmRegistry",
    "DatasetRegistry",
    "MetricRegistry",
    "AlgorithmEntry",
    "MetricEntry",
    "AlgorithmStateEnum",
    "AlgorithmStatusEntry",
    "AlgorithmStatusRegistry",
    "Registry",
]