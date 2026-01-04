import logging
from types import ModuleType

import pytest
from conftest import MockClassIsBase, MockClassNonBase

from streamsight.registries import Registry


logger = logging.getLogger(__name__)


class TestRegistry:
    """Comprehensive test suite for the Registry class."""

    def test_init_registers_from_all(self, registry_valid: Registry) -> None:
        """Test that __init__ registers classes from __all__."""
        assert "MockClassNonBase" in registry_valid.registered
        assert "MockClassIsBase" in registry_valid.registered
        assert registry_valid.registered["MockClassNonBase"] == MockClassNonBase
        assert registry_valid.registered["MockClassIsBase"] == MockClassIsBase
        # Should not register functions
        assert "mock_function" not in registry_valid.registered
        # Should not register constants or other non-class types
        assert "SOME_CONSTANT" not in registry_valid.registered

    def test_init_no_all_attribute(self, registry_without_all: Registry) -> None:
        """Test __init__ when module has no __all__."""
        # Should not register anything if no __all__
        assert len(registry_without_all.registered) == 0

    def test_init_all_with_nonexistent_attr(self, mock_module: ModuleType) -> None:
        """Test __init__ when __all__ contains nonexistent attributes."""
        mock_module.__all__ = ["MockClassNonBase", "NonExistentClass"]
        registry = Registry(mock_module)
        assert "MockClassNonBase" in registry.registered
        assert "NonExistentClass" not in registry.registered

    def test_getitem_registered_key(self, registry_valid: Registry) -> None:
        """Test __getitem__ for registered keys."""
        assert registry_valid["MockClassNonBase"] == MockClassNonBase
        assert registry_valid["MockClassIsBase"] == MockClassIsBase

    def test_getitem_nonexistent_key(self, registry_valid: Registry) -> None:
        """Test __getitem__ raises KeyError for nonexistent keys."""
        with pytest.raises(KeyError):
            _ = registry_valid["NonExistent"]

    def test_contains_registered_key(self, registry_valid: Registry) -> None:
        """Test __contains__ for registered keys."""
        assert "MockClassNonBase" in registry_valid
        assert "MockClassIsBase" in registry_valid

    def test_contains_nonexistent_key(self, registry_valid: Registry) -> None:
        """Test __contains__ for nonexistent keys."""
        assert "NonExistent" not in registry_valid

    def test_get_registered_key(self, registry_valid: Registry) -> None:
        """Test get method for registered keys."""
        assert registry_valid.get("MockClassNonBase") == MockClassNonBase

    def test_get_unregistered_key(self, registry_valid: Registry, mock_module: ModuleType) -> None:
        """Test get method for unregistered keys."""
        mock_module.UnregisteredClass = MockClassNonBase  # type: ignore
        with pytest.raises(KeyError):
            registry_valid.get("UnregisteredClass")

    def test_get_nonexistent_key(self, registry_valid: Registry) -> None:
        """Test get method raises KeyError for nonexistent keys."""
        with pytest.raises(KeyError):
            registry_valid.get("NonExistent")

    def test_register_new_key(self, registry_valid: Registry) -> None:
        """Test registering a new key."""
        registry_valid.register("NewClass", MockClassNonBase)
        assert "NewClass" in registry_valid
        assert registry_valid["NewClass"] == MockClassNonBase
        assert "NewClass" in registry_valid.get_registered_keys()

    def test_register_duplicate_key_raises_error(self, registry_valid: Registry) -> None:
        """Test registering a duplicate key raises KeyError."""
        with pytest.raises(KeyError):
            registry_valid.register("MockClassNonBase", MockClassIsBase)

    def test_register_unregistered_key_that_exists_in_src(self, registry_valid: Registry, mock_module: ModuleType) -> None:
        """Test registering a key that exists in src but not registered."""
        mock_module.ExistingClass = MockClassNonBase  # type: ignore
        registry_valid.register("ExistingClass", MockClassNonBase)
        assert "ExistingClass" in registry_valid.registered

    def test_get_registered_keys(self, registry_valid: Registry) -> None:
        """Test get_registered_keys returns list of registered keys."""
        keys = registry_valid.get_registered_keys()
        assert isinstance(keys, list)
        assert "MockClassNonBase" in keys
        assert len(keys) == 1

    def test_get_registered_keys_empty_registry(self, registry_without_all: Registry) -> None:
        """Test get_registered_keys on empty registry."""
        keys = registry_without_all.get_registered_keys()
        assert keys == []

    def test_register_and_getitem(self, registry_valid: Registry) -> None:
        """Test registering and then retrieving."""
        registry_valid.register("TestClass", MockClassNonBase)
        assert registry_valid["TestClass"] == MockClassNonBase
        assert registry_valid.get("TestClass") == MockClassNonBase

    def test_src_attribute(self, registry_valid: Registry, mock_module: ModuleType) -> None:
        """Test that src attribute is set correctly."""
        assert registry_valid.src == mock_module
