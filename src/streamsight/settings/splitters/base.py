import logging
from abc import ABC, abstractmethod

from streamsight.matrix import InteractionMatrix


logger = logging.getLogger(__name__)


class Splitter(ABC):
    """Abstract base class for dataset splitters.

    Implementations should split an :class:`InteractionMatrix` into two
    parts according to a splitting condition (for example, by timestamp).
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        """Return the class name of the splitter.

        Returns:
            The splitter class name.
        """
        return self.__class__.__name__

    @property
    def identifier(self) -> str:
        """Return a string identifier including the splitter's parameters.

        The identifier includes the class name and a comma-separated list of
        attribute name/value pairs from `self.__dict__`.

        Returns:
            Identifier string like `Name(k1=v1,k2=v2)`.
        """

        paramstring = ",".join((f"{k}={v}" for k, v in self.__dict__.items()))
        return self.name + f"({paramstring})"

    @abstractmethod
    def split(self, data: InteractionMatrix) -> tuple[InteractionMatrix, InteractionMatrix]:
        """Split an interaction matrix into two parts.

        Args:
            data (InteractionMatrix): The interaction dataset to split.

        Returns:
            A pair of `InteractionMatrix` objects representing the two parts.

        Raises:
            NotImplementedError: If the concrete splitter does not implement this method.
        """

        raise NotImplementedError(f"{self.name} must implement the _split method.")
