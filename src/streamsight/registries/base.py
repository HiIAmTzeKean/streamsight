import inspect
import logging
from types import ModuleType
from typing import Generic, TypeVar

from ..models import BaseModel


logger = logging.getLogger(__name__)


T = TypeVar('T', bound=BaseModel)


class Registry(Generic[T], BaseModel):
    """A Registry is a wrapper for a dictionary that maps names to Python types.

    Most often, this is used to map names to classes.
    """

    def __init__(self, src: ModuleType) -> None:
        self.registered: dict[str, type[T]] = {}
        self.src = src
        self._register_all_src()

    def _register_all_src(self) -> None:
        """Register all classes from the src module."""
        if not hasattr(self.src, "__all__"):
            raise AttributeError(f"Source module {self.src} has no __all__ attribute")
        if self.src.__all__ is None:
            raise AttributeError(f"Source module {self.src} has __all__ set to None")
        for class_name in self.src.__all__:
            try:
                cls = getattr(self.src, class_name)
                if not inspect.isclass(cls):
                    continue
                self.register(class_name, cls)
            except AttributeError:
                # Skip if the attribute doesn't exist
                continue

    def __getitem__(self, key: str) -> type[T]:
        """Retrieve the type for the given key.

        Args:
            key: The key of the type to fetch.

        Returns:
            The class type associated with the key.

        Raises:
            KeyError: If the key is not found in the registry.
        """
        try:
            return self.registered[key]
        except KeyError:
            raise KeyError(f"key `{key}` not found in registry")

    def __contains__(self, key: str) -> bool:
        """Check if the given key is known to the registry.

        Args:
            key: The key to check.

        Returns:
            True if the key is known, False otherwise.
        """
        try:
            self[key]
            return True
        except KeyError:
            return False

    def get(self, key: str) -> type[T]:
        """Retrieve the value for this key.

        This value is a Python type, most often a class.

        Args:
            key: The key to fetch.

        Returns:
            The class type associated with the key.

        Raises:
            KeyError: If the key is not found in the registry.
        """
        return self[key]

    def register(self, key: str, cls: type[T]) -> None:
        """Register a new Python type (most often a class).

        After registration, the key can be used to fetch the Python type from the registry.

        Args:
            key: Key to register the type at. Needs to be unique to the registry.
            cls: Class to register.

        Raises:
            KeyError: If the key is already registered.
        """
        if key in self.registered:
            raise KeyError(f"key `{key}` already registered")
        self.registered[key] = cls

    def get_registered_keys(self, include_base: bool = False) -> list[str]:
        """Get a list of all registered keys.

        Returns:
            A list of all registered keys.
        """
        if include_base:
            return list(self.registered.keys())
        else:
            return [key for key, cls in self.registered.items() if not getattr(cls, "IS_BASE", True)]

    def registered_values(self) -> list[type[T]]:
        """Get a list of all registered types.

        Returns:
            A list of all registered types.
        """
        return [self.registered[key] for key in self.get_registered_keys(include_base=False)]

    def registered_items(self) -> list[tuple[str, type[T]]]:
        """Get a list of all registered key-type pairs.

        Returns:
            A list of all registered key-type pairs.
        """
        return [(key, self.registered[key]) for key in self.get_registered_keys(include_base=False)]
