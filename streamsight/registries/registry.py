from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, NamedTuple, Optional
from uuid import UUID

import streamsight.algorithms
import streamsight.datasets
import streamsight.metrics
from streamsight.algorithms import Algorithm


class Registry:
    """
    A Registry is a wrapper for a dictionary that maps
    names to Python types (most often classes).
    """

    def __init__(self, src):
        self.registered: Dict[str, type] = {}
        self.src = src

    def __getitem__(self, key: str) -> type:
        """Retrieve the type for the given key.

        :param key: the key of the type to fetch
        :type key: str
        :returns: The class type associated with the key
        :rtype: type
        """
        if key in self.registered:
            return self.registered[key]
        else:
            return getattr(self.src, key)

    def __contains__(self, key: str) -> bool:
        """Check if the given key is known to the registry.

        :param key: The key to check.
        :type key: str
        :return: True if the key is known
        :rtype: bool
        """
        try:
            self[key]
            return True
        except AttributeError:
            return False

    def get(self, key: str) -> type:
        """Retrieve the value for this key. This value is a Python type (most often a class).

        :param key: The key to fetch
        :type key: str
        :return: The class type associated with the key
        :rtype: type
        """
        return self[key]

    def register(self, key: str, c: type):
        """Register a new Python type (most often a class).

        After registration, the key can be used to fetch the Python type from the registry.

        :param key: key to register the type at. Needs to be unique to the registry.
        :type key: str
        :param c: class to register.
        :type c: type
        """
        if key in self:
            raise KeyError(f"key {key} already registered")
        self.registered[key] = c
        
class AlgorithmRegistry(Registry):
    """Registry for easy retrieval of algorithm types by name.

    The registry comes preregistered with all streamsight algorithms.
    """

    def __init__(self):
        super().__init__(streamsight.algorithms)

class MetricRegistry(Registry):
    """Registry for easy retrieval of metric types by name.

    The registry comes preregistered with all the streamsight metrics.
    """
    def __init__(self):
        super().__init__(streamsight.metrics)

class DatasetRegistry(Registry):
    """Registry for easy retrieval of dataset types by name.

    The registry comes preregistered with all the streamsight datasets.
    """
    def __init__(self):
        super().__init__(streamsight.datasets)
    

ALGORITHM_REGISTRY = AlgorithmRegistry()
"""Registry for algorithms.

Contains the streamsight algorithms by default,
and allows registration of new algorithms via the `register` function.

Example::

    from streamsight.pipelines import ALGORITHM_REGISTRY

    # Construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get('ItemKNN')(K=20)

    from streamsight.algorithms import ItemKNN
    ALGORITHM_REGISTRY.register('HelloWorld', ItemKNN)

    # Also construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get('HelloWorld')(K=20)
"""

METRIC_REGISTRY = MetricRegistry()
"""Registry for metrics.

Contains the streamsight metrics by default,
and allows registration of new metrics via the `register` function.

Example::

    from streamsight.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('Recall')(K=20)

    from streamsight.algorithms import Recall
    METRIC_REGISTRY.register('HelloWorld', Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('HelloWorld')(K=20)

"""


DATASET_REGISTRY = DatasetRegistry()
"""Registry for datasets.

Contains the streamsight metrics by default,
and allows registration of new metrics via the `register` function.

Example::

    from streamsight.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('Recall')(K=20)

    from streamsight.algorithms import Recall
    METRIC_REGISTRY.register('HelloWorld', Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('HelloWorld')(K=20)
"""


class AlgorithmEntry(NamedTuple):
    """Entry for the algorithm registry.
    
    The intended use of this class is to store the name of the algorithm and the
    parameters that the algorithm should take. Mainly this will happen during
    the building phase of the evaluator pipeline in :class:`Builder`.

    :param name: Name of the algorithm
    :type name: str
    :param params: Parameters that do not require optimization as key-value pairs,
        where the key is the name of the hyperparameter and value is the value it should take.
    :type params: Dict[str, Any], optional
    """
    name: str
    params: Optional[Dict[str, Any]] = None


class MetricEntry(NamedTuple):
    """Entry for the metric registry.
    
    The intended use of this class is to store the name of the metric and the
    top K value for the metric specified by the user. Mainly this will happen
    during the building phase of the evaluator pipeline in :class:`Builder`.

    :param name: Name of the algorithm
    :type name: str
    :param K: Top K value for the metric.
    :type K: int, optional
    """
    name: str
    K: Optional[int] = None

class AlgorithmStateEnum(StrEnum):
    """Enum for the state of the algorithm
    
    This enum is used to keep track of the state of the algorithm during
    the streaming process in the :class:`EvaluatorStreamer`.
    """
    NEW = "NEW"
    READY = "READY"
    PREDICTED = "PREDICTED"
    COMPLETED = "COMPLETED"

@dataclass
class AlgorithmStatusEntry():
    """Entry for the algorithm status registry
    
    This entry is used to store the status of the algorithm in
    :class:`AlgorithmStatusRegistry`. The entry contains the name of the
    algorithm, the unique identifier for the algorithm, the state of the
    algorithm, the data segment the algorithm is associated with, and a
    pointer to the algorithm object.
    
    This entry is mainly used during the streaming process in the
    :class:`EvaluatorStreamer` to keep track of the state of the algorithm.

    :param name: Name of the algorithm
    :type name: str
    :param algo_id: Unique identifier for the algorithm
    :type algo_id: UUID
    :param state: State of the algorithm
    :type state: AlgorithmStateEnum
    :param data_segment: Data segment the algorithm is associated with
    :type data_segment: Optional[int]
    :param algo_ptr: Pointer to the algorithm object
    :type algo_ptr: Optional[Any]
    """
    name: str
    algo_id : UUID
    state: AlgorithmStateEnum
    data_segment: Optional[int] = None
    algo_ptr: Optional[Algorithm] = None

class AlgorithmStatusRegistry:
    """
    Registry for algorithm status.
    
    This registry is used to store the status of the algorithms.
    The status of the algorithms are stored in the registry and can be
    accessed by the user. The status of the algorithms are used to keep
    track of the state of the algorithms.
    
    The registry uses the :class:`AlgorithmStatusEntry` to store the
    status of the algorithms. The registry allows the user to update the
    status of the algorithms and get the status of the algorithms.
    
    The recommended way to use this registry is to use provided function :meth:`update`
    to update the status of the algorithms. Aggregating the status of the
    algorithms can be done by using the :meth:`all_algo_states` function.
    """
    def __init__(self):
        self.registered: Dict[UUID, AlgorithmStatusEntry] = {}
        self.status_counts = {i:0 for i in AlgorithmStateEnum}

    def __iter__(self):
        return iter(self.registered)
    
    def __getitem__(self, key: UUID) -> AlgorithmStatusEntry:
        if key not in self.registered:
            raise AttributeError(f"Algorithm with ID:{key} not registered")
        return self.registered[key]

    def __setitem__(self, key: UUID, entry: AlgorithmStatusEntry):
        if key in self:
            raise KeyError(f"Algorithm with ID:{key} already registered")
        self.registered[key] = entry
    
    def __contains__(self, key: UUID) -> bool:
        """Check if the given key is known to the registry.

        :param key: The key to check.
        :type key: str
        :return: True if the key is known
        :rtype: bool
        """
        try:
            self[key]
            return True
        except AttributeError:
            return False

    def get(self, algo_id: UUID) -> AlgorithmStatusEntry:
        return self[algo_id]

    def register(self, algo_id: UUID, entry: AlgorithmStatusEntry) -> None:
        self[algo_id] = entry
    
    def update(self, algo_id: UUID, state: AlgorithmStateEnum, data_segment: Optional[int] = None) -> None:
        self.status_counts[self[algo_id].state] -= 1
        self[algo_id].state = state
        self.status_counts[state] += 1
        
        if state == AlgorithmStateEnum.READY:
            if data_segment is None:
                raise ValueError(f"Data segment not provided for {AlgorithmStateEnum.READY} state")
            self[algo_id].data_segment = data_segment
    
    def is_all_predicted(self) -> bool:
        return self.status_counts[AlgorithmStateEnum.PREDICTED] == len(self.registered)
    
    def all_algo_states(self) -> Dict[str, AlgorithmStateEnum]:
        states = {}
        for key in self:
            states[f"{self[key].name}_{key}"] = self[key].state
        return states
    
    def set_all_ready(self, data_segment: int) -> None:
        for key in self:
            self.update(key, AlgorithmStateEnum.READY, data_segment)
    
    def get_algorithm_identifier(self, algo_id: UUID) -> str:
        return f"{self[algo_id].name}_{algo_id}"