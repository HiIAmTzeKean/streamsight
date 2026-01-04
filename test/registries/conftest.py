from types import ModuleType

import pytest

from streamsight.models import BaseModel
from streamsight.registries import Registry


class MockClassNonBase(BaseModel):
    IS_BASE = False


class MockClassIsBase:
    pass


def mock_function() -> None:
    pass


def create_mock_module(name: str, all_list: list[str], attributes: dict) -> ModuleType:
    """Helper function to create a mock module with specified attributes."""
    module = ModuleType(name)
    module.__all__ = all_list
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def mock_module() -> ModuleType:
    """Create a mock module for testing Registry."""
    attributes = {
        "MockClassNonBase": MockClassNonBase,
        "MockClassIsBase": MockClassIsBase,
        "mock_function": mock_function,
        "SOME_CONSTANT": 42,
    }
    return create_mock_module(
        "mock_module",
        [
            "MockClassNonBase",
            "MockClassIsBase",
            "mock_function",
            "SOME_CONSTANT",
        ],
        attributes,
    )


@pytest.fixture
def mock_module_no_all() -> ModuleType:
    """Create a mock module without __all__ for testing."""
    attributes = {
        "MockClassNonBase": MockClassNonBase,
    }
    return create_mock_module(
        "mock_module_no_all",
        [],
        attributes,
    )


@pytest.fixture
def registry_valid(mock_module: ModuleType) -> Registry:
    """Create a Registry instance with mock module."""
    return Registry(mock_module)


@pytest.fixture
def registry_without_all(mock_module_no_all: ModuleType) -> Registry:
    """Create a Registry instance with mock module without __all__."""
    return Registry(mock_module_no_all)
