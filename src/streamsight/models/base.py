from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Base class for all streamsight components.

    Provides common properties like name and universal IS_BASE flag.
    """

    IS_BASE: bool = True

    @property
    def name(self) -> str:
        """Name of the object's class.

        :return: Name of the object's class
        :rtype: str
        """
        return self.__class__.__name__


class ParamMixin(ABC):
    """Mixin class for all streamsight components with parameters.

    Provides common properties like name, params, and identifier.
    """
    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get the parameters of the object.

        :return: Parameters of the object
        :rtype: dict
        """
        ...

    @property
    def params(self) -> dict[str, Any]:
        """Parameters of the object.

        :return: Parameters of the object
        :rtype: dict
        """
        return self.get_params()

    @property
    def identifier(self) -> str:
        """Identifier of the object.

        Identifier is made by combining the class name with the parameters
        passed at construction time.

        Constructed by recreating the initialisation call.
        Example: `Algorithm(param_1=value)`

        :return: Identifier of the object
        """
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"
