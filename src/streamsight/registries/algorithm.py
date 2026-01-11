import importlib
import logging
import uuid
from typing import Any, NamedTuple

from .base import Registry


logger = logging.getLogger(__name__)


class AlgorithmRegistry(Registry):
    """Registry for easy retrieval of algorithm types by name.

    The registry is pre-registered with all streamsight algorithms.
    """

    def __init__(self) -> None:
        """Initialize the algorithm registry.

        The registry is initialized with the `streamsight.algorithms` module
        so that all built-in algorithms are available by default.
        """
        module = importlib.import_module("streamsight.algorithms")
        super().__init__(module)


ALGORITHM_REGISTRY = AlgorithmRegistry()
"""Registry instantiation for algorithms.

Contains the streamsight algorithms by default and allows registration of
new algorithms via the `register` function.

Examples:
    ```python
    from streamsight.pipelines import ALGORITHM_REGISTRY

    # Construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get("ItemKNN")(K=20)

    from streamsight.algorithms import ItemKNN

    ALGORITHM_REGISTRY.register("HelloWorld", ItemKNN)

    # Also construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get("HelloWorld")(K=20)
    ```
"""


class AlgorithmEntry(NamedTuple):
    """Entry for the algorithm registry.

    The intended use of this class is to store the name of the algorithm and
    the parameters that the algorithm should take. Mainly this is used during
    the building phase of the evaluator pipeline in `Builder`.

    Attributes:
        name: Name of the algorithm.
        params: Parameters that do not require optimization as key-value
            pairs, where the key is the hyperparameter name and the value is
            the value it should take.
    """

    name: str
    uuid: uuid.UUID | None = None
    params: None | dict[str, Any] = None
