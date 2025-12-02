import logging
from types import ModuleType

import pytest
from conftest import MockClass1, MockClass2

from streamsight.registries.base import Registry


logger = logging.getLogger(__name__)


class TestRegistry:
    """Comprehensive test suite for the Registry class."""

    def test_init_registers_from_all(self, mock_module: ModuleType) -> None:
        """Test that __init__ registers classes from __all__."""
        registry = Registry(mock_module)
        assert "MockClass1" in registry.registered
        assert "MockClass2" in registry.registered
        assert registry.registered["MockClass1"] == MockClass1
        assert registry.registered["MockClass2"] == MockClass2
        # Should not register functions
        assert "mock_function" not in registry.registered
        # Should not register constants or other non-class types
        assert "SOME_CONSTANT" not in registry.registered

    def test_init_no_all_attribute(self, mock_module_no_all: ModuleType) -> None:
        """Test __init__ when module has no __all__."""
        registry = Registry(mock_module_no_all)
        # Should not register anything if no __all__
        assert len(registry.registered) == 0

    def test_init_all_with_nonexistent_attr(self, mock_module: ModuleType) -> None:
        """Test __init__ when __all__ contains nonexistent attributes."""
        mock_module.__all__ = ["MockClass1", "NonExistentClass"]
        registry = Registry(mock_module)
        assert "MockClass1" in registry.registered
        assert "NonExistentClass" not in registry.registered

    def test_getitem_registered_key(self, registry: Registry) -> None:
        """Test __getitem__ for registered keys."""
        assert registry["MockClass1"] == MockClass1
        assert registry["MockClass2"] == MockClass2

    def test_getitem_nonexistent_key(self, registry: Registry) -> None:
        """Test __getitem__ raises KeyError for nonexistent keys."""
        with pytest.raises(KeyError):
            _ = registry["NonExistent"]

    def test_contains_registered_key(self, registry: Registry) -> None:
        """Test __contains__ for registered keys."""
        assert "MockClass1" in registry
        assert "MockClass2" in registry

    def test_contains_nonexistent_key(self, registry: Registry) -> None:
        """Test __contains__ for nonexistent keys."""
        assert "NonExistent" not in registry

    def test_get_registered_key(self, registry: Registry) -> None:
        """Test get method for registered keys."""
        assert registry.get("MockClass1") == MockClass1

    def test_get_unregistered_key(self, registry: Registry, mock_module: ModuleType) -> None:
        """Test get method for unregistered keys."""
        mock_module.UnregisteredClass = MockClass1
        with pytest.raises(KeyError):
            registry.get("UnregisteredClass")

    def test_get_nonexistent_key(self, registry: Registry) -> None:
        """Test get method raises KeyError for nonexistent keys."""
        with pytest.raises(KeyError):
            registry.get("NonExistent")

    def test_register_new_key(self, registry: Registry) -> None:
        """Test registering a new key."""
        registry.register("NewClass", MockClass1)
        assert "NewClass" in registry
        assert registry["NewClass"] == MockClass1
        assert "NewClass" in registry.get_registered_keys()

    def test_register_duplicate_key_raises_error(self, registry: Registry) -> None:
        """Test registering a duplicate key raises KeyError."""
        with pytest.raises(KeyError):
            registry.register("MockClass1", MockClass2)

    def test_register_unregistered_key_that_exists_in_src(self, registry: Registry, mock_module: ModuleType) -> None:
        """Test registering a key that exists in src but not registered."""
        mock_module.ExistingClass = MockClass1
        registry.register("ExistingClass", MockClass1)
        assert "ExistingClass" in registry.registered

    def test_get_registered_keys(self, registry: Registry) -> None:
        """Test get_registered_keys returns list of registered keys."""
        keys = registry.get_registered_keys()
        assert isinstance(keys, list)
        assert "MockClass1" in keys
        assert "MockClass2" in keys
        assert len(keys) == 2

    def test_get_registered_keys_empty_registry(self, mock_module_no_all: ModuleType) -> None:
        """Test get_registered_keys on empty registry."""
        registry = Registry(mock_module_no_all)
        keys = registry.get_registered_keys()
        assert keys == []

    def test_register_and_getitem(self, registry: Registry) -> None:
        """Test registering and then retrieving."""
        registry.register("TestClass", MockClass1)
        assert registry["TestClass"] == MockClass1
        assert registry.get("TestClass") == MockClass1

    def test_src_attribute(self, registry: Registry, mock_module: ModuleType) -> None:
        """Test that src attribute is set correctly."""
        assert registry.src == mock_module
