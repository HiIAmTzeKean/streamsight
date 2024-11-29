from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple, Optional
from uuid import UUID

import streamsight.algorithms
from streamsight.algorithms import Algorithm
from .base import Registry


class AlgorithmRegistry(Registry):
    """Registry for easy retrieval of algorithm types by name.

    The registry is pre-registered with all streamsight algorithms.
    """

    def __init__(self) -> None:
        """Initialize the algorithm registry.

        The registry is initialized with the `streamsight.algorithms` module
        so that all built-in algorithms are available by default.
        """
        super().__init__(streamsight.algorithms)


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
    params: Optional[dict[str, Any]] = None


class AlgorithmStateEnum(StrEnum):
    """Enum for the state of the algorithm.

    Used to keep track of the state of the algorithm during the streaming
    process in the `EvaluatorStreamer`.
    """

    NEW = "NEW"
    READY = "READY"
    PREDICTED = "PREDICTED"
    COMPLETED = "COMPLETED"


@dataclass
class AlgorithmStatusEntry:
    """Entry for the algorithm status registry.

    This dataclass stores the status of an algorithm for use by
    `AlgorithmStatusRegistry`. It contains the algorithm name, unique
    identifier, current state, associated data segment, and an optional
    pointer to the algorithm object.

    Attributes:
        name: Name of the algorithm.
        algo_id: Unique identifier for the algorithm.
        state: State of the algorithm.
        data_segment: Data segment the algorithm is associated with.
        algo_ptr: Pointer to the algorithm object.
    """

    name: str
    algo_id: UUID
    state: AlgorithmStateEnum
    data_segment: None | int = None
    algo_ptr: None | Algorithm = None


class AlgorithmStatusRegistry:
    """Registry for algorithm status.

    Maintains a mapping from algorithm UUIDs to :class:`AlgorithmStatusEntry`
    objects and keeps counts of entries per :class:`AlgorithmStateEnum`.

    Attributes:
        registered: Mapping from algorithm UUID to its status entry.
        status_counts: Number of entries currently in each state.

    Raises:
        AttributeError: Accessing an unregistered UUID via `__getitem__` or
            `get`.
        KeyError: Attempting to register an algorithm UUID that is already
            present.
        ValueError: Calling :meth:`update` with
            `state == AlgorithmStateEnum.READY` without providing a
            `data_segment`.

    Example:
        ```python
        from uuid import uuid4
        reg = AlgorithmStatusRegistry()
        entry = AlgorithmStatusEntry(
            name="ItemKNN", algo_id=uuid4(), state=AlgorithmStateEnum.NEW
        )
        reg.register(entry.algo_id, entry)
        reg.update(entry.algo_id, AlgorithmStateEnum.READY, data_segment=0)
        reg.is_all_predicted()
        # => False
        ```
    """

    def __init__(self) -> None:
        """Initialize the status registry.

        Initializes the internal mapping of registered algorithm status
        entries and a counter for entries grouped by their state.
        """
        self.registered: dict[UUID, AlgorithmStatusEntry] = {}
        self.status_counts = {i: 0 for i in AlgorithmStateEnum}

    def __iter__(self) -> Iterator[UUID]:
        """Return an iterator over registered algorithm UUIDs.

        Returns:
            An iterator over the UUIDs of registered entries.
        """
        return iter(self.registered)

    def __getitem__(self, key: UUID) -> AlgorithmStatusEntry:
        """Return the status entry for `key`.

        Args:
            key: The UUID of the algorithm to retrieve.

        Returns:
            The status entry associated with `key`.

        Raises:
            AttributeError: If `key` is not registered.
        """
        if key not in self.registered:
            raise AttributeError(f"Algorithm with ID:{key} not registered")
        return self.registered[key]

    def __setitem__(self, key: UUID, entry: AlgorithmStatusEntry) -> None:
        """Register a new algorithm status entry under `key`.

        Args:
            key: The UUID to register the entry under.
            entry: The status entry to register.

        Raises:
            KeyError: If `key` is already registered.
        """
        if key in self:
            raise KeyError(f"Algorithm with ID:{key} already registered")
        self.registered[key] = entry

    def __contains__(self, key: UUID) -> bool:
        """Return whether the given key is known to the registry.

        Args:
            key: The key to check.

        Returns:
            True if the key is registered, False otherwise.
        """
        try:
            self[key]
            return True
        except AttributeError:
            return False

    def get(self, algo_id: UUID) -> AlgorithmStatusEntry:
        """Get the :class:`AlgorithmStatusEntry` for `algo_id`.

        This is a convenience alias for `__getitem__`.

        Args:
            algo_id: Algorithm UUID to retrieve.

        Returns:
            The status entry for `algo_id`.

        Raises:
            AttributeError: If `algo_id` is not registered.
        """
        return self[algo_id]

    def register(self, algo_id: UUID, entry: AlgorithmStatusEntry) -> None:
        """Register a new algorithm status entry.

        This is a convenience alias for `__setitem__`.

        Args:
            algo_id: UUID to register the entry under.
            entry: Entry to register.

        Raises:
            KeyError: If `algo_id` is already registered.
        """
        self[algo_id] = entry

    def update(
        self, algo_id: UUID, state: AlgorithmStateEnum, data_segment: None | int = None
    ) -> None:
        """Update the state (and optional data segment) of a registered entry.

        Args:
            algo_id: UUID of the algorithm to update.
            state: New state to assign.
            data_segment: Data segment to associate with the algorithm when
                transitioning to `READY`.

        Raises:
            AttributeError: If `algo_id` is not registered.
            ValueError: If `state` is `AlgorithmStateEnum.READY` and
                `data_segment` is `None`.
        """
        if algo_id not in self.registered:
            raise AttributeError(f"Algorithm with ID:{algo_id} not registered")

        # decrement previous state count
        self.status_counts[self[algo_id].state] -= 1

        # update state and increment new state count
        self[algo_id].state = state
        self.status_counts[state] += 1

        if state == AlgorithmStateEnum.READY:
            if data_segment is None:
                raise ValueError(
                    f"Data segment not provided for {AlgorithmStateEnum.READY} state"
                )
            self[algo_id].data_segment = data_segment

    def is_all_predicted(self) -> bool:
        """Return whether every registered algorithm is in PREDICTED state.

        Returns:
            True if all registered entries have state
            `AlgorithmStateEnum.PREDICTED`, False otherwise.
        """
        return self.status_counts[AlgorithmStateEnum.PREDICTED] == len(
            self.registered
        )

    def is_all_same_data_segment(self) -> bool:
        """Return whether all registered entries share the same data segment.

        Returns:
            True if there is exactly one distinct data segment across all
            registered entries, False otherwise.
        """
        data_segments: set[None | int] = set()
        for key in self:
            data_segments.add(self[key].data_segment)
        return len(data_segments) == 1

    def all_algo_states(self) -> dict[str, AlgorithmStateEnum]:
        """Return a mapping of identifier strings to algorithm states.

        The identifier used is "{name}_{uuid}" for each registered entry.

        Returns:
            Mapping from identifier string to the entry's
            :class:`AlgorithmStateEnum`.
        """
        states: dict[str, AlgorithmStateEnum] = {}
        for key in self:
            states[f"{self[key].name}_{key}"] = self[key].state
        return states

    def set_all_ready(self, data_segment: int) -> None:
        """Set all registered algorithms to the READY state.

        Args:
            data_segment: Data segment to assign to every algorithm.
        """
        for key in self:
            self.update(key, AlgorithmStateEnum.READY, data_segment)

    def get_algorithm_identifier(self, algo_id: UUID) -> str:
        """Return a stable identifier string for the algorithm.

        Args:
            algo_id: UUID of the algorithm.

        Returns:
            Identifier in the format "{name}_{uuid}".

        Raises:
            AttributeError: If `algo_id` is not registered.
        """
        return f"{self[algo_id].name}_{algo_id}"
