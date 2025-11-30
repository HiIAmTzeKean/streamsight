from types import ModuleType

import pytest

from streamsight.registries.base import Registry


class MockClass1:
    pass


class MockClass2:
    pass


def mock_function() -> None:
    pass


@pytest.fixture
def mock_module() -> ModuleType:
    """Create a mock module for testing Registry."""
    module = ModuleType("mock_module")
    module.__all__ = ["MockClass1", "MockClass2", "mock_function", "SOME_CONSTANT"]
    module.MockClass1 = MockClass1
    module.MockClass2 = MockClass2
    module.mock_function = mock_function
    module.SOME_CONSTANT = 42
    return module


@pytest.fixture
def mock_module_no_all() -> ModuleType:
    """Create a mock module without __all__ for testing."""
    module = ModuleType("mock_module_no_all")
    module.__all__ = []
    module.MockClass1 = MockClass1
    return module


@pytest.fixture
def registry(mock_module: ModuleType) -> Registry:
    """Create a Registry instance with mock module."""
    return Registry(mock_module)
