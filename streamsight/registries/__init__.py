"""
.. currentmodule:: streamsight.registries

Registry
----------------

Registry for algorithms, metrics and datasets. These registries are used to store
the algorithms, metrics and datasets that are used in the experiments. The registries
aid the programmer to keep track of valid classes and to easily instantiate them.

Contains the streamsight algorithms by default, and allows registration of
new algorithms via the `register` function.

Example::
    
        from streamsight.pipelines import ALGORITHM_REGISTRY
        from streamsight.algorithms import ItemKNNStatic
    
        algo = ALGORITHM_REGISTRY.get('ItemKNNStatic')(K=10)
        ALGORITHM_REGISTRY.register('algo_1', ItemKNNStatic)

.. autosummary::
    :toctree: generated/

    ALGORITHM_REGISTRY
    DATASET_REGISTRY
    METRIC_REGISTRY
    AlgorithmRegistry
    DatasetRegistry
    MetricRegistry
    
Entries
----------------

Entries for algorithms and metrics. These entries are used to store the algorithms
and metrics in the registries. The entries are used to store the class and the
parameters that are used to instantiate the class. These entires will be used
in :class:`EvaluatorPipeline` to instantiate the algorithms and metrics.

.. autosummary::
    :toctree: generated/
    
    AlgorithmEntry
    MetricEntry
    
Status Registry
----------------

Registry for algorithm status. This registry is used to store the status of the
algorithms. The status of the algorithms are stored in the registry and can be
accessed by the user. The status of the algorithms are used to keep track of the
state of the algorithms.

.. autosummary::
    :toctree: generated/
    
    AlgorithmStateEnum
    AlgorithmStatusEntry
    AlgorithmStatusRegistry

"""

from streamsight.registries.registry import (ALGORITHM_REGISTRY,
                                             DATASET_REGISTRY, METRIC_REGISTRY,
                                             AlgorithmRegistry,
                                             DatasetRegistry, MetricRegistry,
                                             AlgorithmEntry, MetricEntry,
                                             AlgorithmStateEnum,
                                             AlgorithmStatusEntry,
                                             AlgorithmStatusRegistry)
