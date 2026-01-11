import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

from streamsight.algorithms import Algorithm
from ..utils.uuid_util import generate_algorithm_uuid


logger = logging.getLogger(__name__)


class AlgorithmStateEnum(StrEnum):
    """Enum for the state of the algorithm.

    Used to keep track of the state of the algorithm during the streaming
    process in the `EvaluatorStreamer`.
    """

    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    PREDICTED = "PREDICTED"
    COMPLETED = "COMPLETED"


@dataclass
class AlgorithmStateEntry:
    """Entry for the algorithm status registry.

    This dataclass stores the status of an algorithm for use by
    `AlgorithmStateManager`. It contains the algorithm name, unique
    identifier, current state, associated data segment, and an optional
    pointer to the algorithm object.

    Attributes:
        name: Name of the algorithm.
        algo_uuid: Unique identifier for the algorithm.
        state: State of the algorithm.
        data_segment: Data segment the algorithm is associated with.
        params: Parameters for the algorithm.
        algo_ptr: Pointer to the algorithm object.
    """

    name: str
    algo_uuid: UUID
    state: AlgorithmStateEnum = AlgorithmStateEnum.NEW
    data_segment: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    algo_ptr: None | type[Algorithm] | Algorithm = None


class AlgorithmStateManager:
    def __init__(self) -> None:
        self._algorithms: dict[UUID, AlgorithmStateEntry] = {}

    def __iter__(self) -> Iterator[UUID]:
        """Return an iterator over registered algorithm UUIDs.

        Allows iteration over the UUIDs of registered entries.

        Returns:
            An iterator over the UUIDs of registered entries.
        """
        return iter(self._algorithms)

    def __len__(self) -> int:
        """Return the number of registered algorithms.

        Returns:
            The number of registered algorithms.
        """
        return len(self._algorithms)

    def values(self) -> Iterator[AlgorithmStateEntry]:
        """Return an iterator over registered AlgorithmStateEntry objects.

        Allows iteration over the registered entries.

        Returns:
            An iterator over the registered entries.
        """
        return iter(self._algorithms.values())

    def __getitem__(self, key: UUID) -> AlgorithmStateEntry:
        if key not in self._algorithms:
            raise ValueError(f"Algorithm with ID:{key} not registered")
        return self._algorithms[key]

    def __setitem__(self, key: UUID, entry: AlgorithmStateEntry) -> None:
        """Register a new algorithm status entry under `key`.

        Allows the use of square bracket notation to register new entries.

        Args:
            key: The UUID to register the entry under.
            entry: The status entry to register.

        Raises:
            KeyError: If `key` is already registered.
        """
        if key in self:
            raise KeyError(f"Algorithm with ID:{key} already registered")
        self._algorithms[key] = entry

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

    def get(self, algo_id: UUID) -> AlgorithmStateEntry:
        """Get the :class:`AlgorithmStateEntry` for `algo_id`."""
        return self[algo_id]

    def get_state(self, algo_id: UUID) -> AlgorithmStateEnum:
        """Get the current state of the algorithm with `algo_id`."""
        return self[algo_id].state

    def register(
        self,
        name: None | str = None,
        algo_ptr: None | type[Algorithm] | Algorithm = None,
        params: dict[str, Any] = {},
        algo_uuid: None | UUID = None,
    ) -> UUID:
        """Register new algorithm"""
        if not name and not algo_ptr:
            raise ValueError("Either name or algo_ptr must be provided for registration")
        elif algo_ptr and isinstance(algo_ptr, type):
            algo_ptr = algo_ptr(**params)
            name = name or algo_ptr.identifier
        elif algo_ptr and hasattr(algo_ptr, "identifier") and not name:
            name = name or algo_ptr.identifier  # type: ignore[attr-defined]
        elif not name:
            # This should not happen if name was provided or algo_ptr has identifier
            raise ValueError("Algorithm name was not provided and could not be inferred from Algorithm pointer")

        if algo_uuid is None:
            algo_uuid = generate_algorithm_uuid(name)

        entry = AlgorithmStateEntry(algo_uuid=algo_uuid, name=name, algo_ptr=algo_ptr, params=params)
        self._algorithms[algo_uuid] = entry
        logger.info(f"Registered algorithm '{name}' with ID {algo_uuid}")
        return algo_uuid

    def can_request_training_data(self, algo_id: UUID) -> tuple[bool, str]:
        """Check if algorithm can request training data"""
        if algo_id not in self._algorithms:
            return False, f"Algorithm {algo_id} not registered"

        state = self._algorithms[algo_id].state

        if state == AlgorithmStateEnum.COMPLETED:
            return False, "Algorithm has completed evaluation"
        if state == AlgorithmStateEnum.NEW:
            return False, "The algorithm must be set to READY state first"
        if state == AlgorithmStateEnum.PREDICTED:
            return False, "Algorithm has already requested data for this window"
        if state == AlgorithmStateEnum.READY:
            return True, ""

        return False, f"Unknown state {state}"

    def can_request_unlabeled_data(self, algo_id: UUID) -> tuple[bool, str]:
        """Check if algorithm can request unlabeled data"""
        if algo_id not in self._algorithms:
            return False, f"Algorithm {algo_id} not registered"

        state = self._algorithms[algo_id].state

        if state == AlgorithmStateEnum.RUNNING:
            return True, ""
        if state == AlgorithmStateEnum.COMPLETED:
            return False, "Algorithm has completed evaluation"
        if state == AlgorithmStateEnum.NEW:
            return False, "The algorithm must be set to RUNNING state to request unlabeled data"
        if state == AlgorithmStateEnum.PREDICTED:
            return False, "Algorithm has already requested data for this window"
        if state == AlgorithmStateEnum.READY:
            return (
                False,
                "The algorithm must be set to RUNNING state to request unlabeled data. Request training data first",
            )

        return False, f"Unknown state {state}"

    def can_submit_prediction(self, algo_id: UUID) -> tuple[bool, str]:
        """Check if algorithm can submit prediction"""
        if algo_id not in self._algorithms:
            return False, f"Algorithm {algo_id} not registered"

        state = self._algorithms[algo_id].state

        if state == AlgorithmStateEnum.RUNNING:
            return True, ""
        if state == AlgorithmStateEnum.READY:
            return False, "There is new data to be requested"
        if state == AlgorithmStateEnum.NEW:
            return False, "Algorithm must request data first"
        if state == AlgorithmStateEnum.PREDICTED:
            return False, "Algorithm already submitted prediction for this window"
        if state == AlgorithmStateEnum.COMPLETED:
            return False, "Algorithm has completed evaluation"

        return False, f"Unknown state {state}"

    def transition(self, algo_id: UUID, new_state: AlgorithmStateEnum, data_segment: None | int = None) -> None:
        """Transition algorithm to new state with validation"""
        if algo_id not in self._algorithms:
            raise ValueError(f"Algorithm {algo_id} not registered")

        entry = self._algorithms[algo_id]
        old_state = entry.state

        # Define valid transitions
        valid_transitions = {
            # old_state: [list of valid new_states]
            AlgorithmStateEnum.NEW: [AlgorithmStateEnum.READY, AlgorithmStateEnum.COMPLETED],
            AlgorithmStateEnum.READY: [AlgorithmStateEnum.RUNNING],
            AlgorithmStateEnum.RUNNING: [AlgorithmStateEnum.PREDICTED],
            AlgorithmStateEnum.PREDICTED: [AlgorithmStateEnum.READY, AlgorithmStateEnum.COMPLETED],
            AlgorithmStateEnum.COMPLETED: [],
        }

        if new_state not in valid_transitions.get(old_state, []):
            raise ValueError(f"Invalid transition: {old_state} -> {new_state}")

        entry.state = new_state
        if data_segment is not None:
            entry.data_segment = data_segment

        logger.debug(f"Algorithm '{entry.name}' transitioned {old_state.value} -> {new_state.value}")

    def is_all_predicted(self) -> bool:
        """Return whether every registered algorithm is in PREDICTED state.

        Returns:
            True if all registered entries have state
            `AlgorithmStateEnum.PREDICTED`, False otherwise.
        """
        if not self._algorithms:
            return False
        return all(entry.state == AlgorithmStateEnum.PREDICTED for entry in self._algorithms.values())

    def get_all_states(self) -> dict[str, AlgorithmStateEnum]:
        """Get state of all algorithms"""
        return {entry.name: entry.state for entry in self._algorithms.values()}

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
            self.transition(key, AlgorithmStateEnum.READY, data_segment)

    def get_algorithm_identifier(self, algo_id: UUID) -> str:
        """Return a stable identifier string for the algorithm.

        Args:
            algo_id: UUID of the algorithm.

        Returns:
            Identifier in the format "{name}_{uuid}".

        Raises:
            AttributeError: If `algo_id` is not registered.
        """
        if algo_id not in self._algorithms:
            raise AttributeError(f"Algorithm with ID:{algo_id} not registered")
        return f"{self[algo_id].name}_{algo_id}"
