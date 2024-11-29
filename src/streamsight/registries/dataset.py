import streamsight.datasets
from .base import Registry


class DatasetRegistry(Registry):
    """Registry for easy retrieval of dataset types by name.

    The registry comes preregistered with all the streamsight datasets.
    """

    def __init__(self) -> None:
        super().__init__(streamsight.datasets)


DATASET_REGISTRY = DatasetRegistry()
"""Registry for datasets.

Contains the streamsight metrics by default,
and allows registration of new metrics via the `register` function.

Example:
    ```python
    from streamsight.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('Recall')(K=20)

    from streamsight.algorithms import Recall
    METRIC_REGISTRY.register('HelloWorld', Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('HelloWorld')(K=20)
    ```
"""
