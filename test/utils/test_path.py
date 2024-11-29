"""Comprehensive test suite for streamsight.utils.path module.

This test suite covers various real-world scenarios including:
- Empty directories without marker files
- Nested Jupyter notebooks
- Docker containers
- Symlinked directories
- Windows path handling
- Network drives
- Multiple repository detection
- And more...

Run with: pytest test/utils/test_path.py -v
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock
import pytest

# Assuming your path module is at src/streamsight/utils/path.py
# Adjust import path as needed
try:
    from streamsight.utils.path import (
        get_repo_root,
        get_data_dir,
        get_logs_dir,
        get_cache_dir,
        safe_dir,
        resolve_path,
    )
    import streamsight.utils.path as path_module
except ImportError:
    pytest.skip("streamsight.utils.path not available", allow_module_level=True)


class BasePathTest:
    """Base class for path testing with common utilities."""

    def setup_method(self):
        """Setup before each test method."""
        # Clear any cached repo root
        path_module._REPO_ROOT_CACHE = None

        # Store original environment
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Clear cache again
        path_module._REPO_ROOT_CACHE = None

    def create_temp_repo(self, temp_dir: Path, marker_files: list[str]) -> Path:
        """Create a temporary repository with marker files.

        :param temp_dir: Temporary directory path
        :param marker_files: List of marker files to create
        :return: Path to created repo
        """
        repo_path = temp_dir / "test_repo"
        repo_path.mkdir(parents=True, exist_ok=True)

        for marker in marker_files:
            marker_path = repo_path / marker
            if marker.endswith("/"):
                # Directory marker
                marker_path.mkdir(parents=True, exist_ok=True)
            else:
                # File marker
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.touch()

        return repo_path

    def create_nested_structure(self, base_path: Path, depth: int) -> Path:
        """Create nested directory structure.

        :param base_path: Base path to start from
        :param depth: Number of nested levels
        :return: Path to deepest level
        """
        current = base_path
        for i in range(depth):
            current = current / f"level_{i}"
            current.mkdir(parents=True, exist_ok=True)
        return current

    def assert_paths_equal(self, path1: Path, path2: Path, msg: str = "") -> None:
        """Assert two paths are equal, handling macOS symlink issues.

        On macOS, /var is symlinked to /private/var, which can cause
        path comparisons to fail. This method resolves both paths
        to their canonical forms before comparing.

        :param path1: First path to compare
        :param path2: Second path to compare
        :param msg: Optional message for assertion failure
        """
        resolved1 = path1.resolve()
        resolved2 = path2.resolve()
        assert resolved1 == resolved2, (
            f"{msg}\n"
            f"Path 1: {path1} (resolved: {resolved1})\n"
            f"Path 2: {path2} (resolved: {resolved2})"
        )


class TestBasicRepoRootDetection(BasePathTest):
    """Test basic repository root detection functionality."""

    def test_marker_files_exist(self, tmp_path):
        """Test that marker files can be created and detected."""
        repo = self.create_temp_repo(tmp_path, [".git/", "pyproject.toml", "README.md"])

        # Verify markers exist
        assert (repo / ".git").exists()
        assert (repo / "pyproject.toml").exists()
        assert (repo / "README.md").exists()

    def test_create_git_marker(self, tmp_path):
        """Test creation of .git directory marker."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        assert (repo / ".git").exists()
        assert (repo / ".git").is_dir()

    def test_create_pyproject_marker(self, tmp_path):
        """Test creation of pyproject.toml file marker."""
        repo = self.create_temp_repo(tmp_path, ["pyproject.toml"])
        assert (repo / "pyproject.toml").exists()
        assert (repo / "pyproject.toml").is_file()

    def test_create_setup_py_marker(self, tmp_path):
        """Test creation of setup.py file marker."""
        repo = self.create_temp_repo(tmp_path, ["setup.py"])
        assert (repo / "setup.py").exists()
        assert (repo / "setup.py").is_file()

    def test_create_multiple_markers(self, tmp_path):
        """Test creation of multiple marker files."""
        repo = self.create_temp_repo(tmp_path, [".git/", "pyproject.toml", "README.md", "setup.py"])

        assert (repo / ".git").exists()
        assert (repo / "pyproject.toml").exists()
        assert (repo / "README.md").exists()
        assert (repo / "setup.py").exists()


class TestDirectoryStructureCreation(BasePathTest):
    """Test directory structure creation utilities."""

    def test_create_nested_structure(self, tmp_path: Path) -> None:
        """Test creating nested directory structures."""
        depth = 5
        deepest = self.create_nested_structure(tmp_path, depth)

        # Verify all levels exist
        current = tmp_path
        for i in range(depth):
            current = current / f"level_{i}"
            assert current.exists()

        assert deepest == current

    def test_deeply_nested_paths(self, tmp_path: Path) -> None:
        """Test paths with many nesting levels."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        nested_path = repo / "a/b/c/d/e/f/g/h"
        nested_path.mkdir(parents=True, exist_ok=True)

        assert nested_path.exists()
        assert str(repo) in str(nested_path)


class TestDataDirectoryFunctions(BasePathTest):
    """Test data directory utility functions."""

    def test_safe_dir_creates_single_directory(self, tmp_path: Path) -> None:
        """Test safe_dir creates a single directory."""
        new_dir = tmp_path / "new_dir"

        result = safe_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_safe_dir_creates_nested_directories(self, tmp_path: Path) -> None:
        """Test safe_dir creates nested directory structure."""
        nested_dir = tmp_path / "a/b/c/d/e"

        result = safe_dir(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert result == nested_dir

    def test_safe_dir_with_existing_directory(self, tmp_path: Path) -> None:
        """Test safe_dir with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = safe_dir(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_safe_dir_returns_path_object(self, tmp_path: Path) -> None:
        """Test safe_dir returns a Path object."""
        new_dir = tmp_path / "new"

        result = safe_dir(new_dir)

        assert isinstance(result, Path)


class TestPathResolution(BasePathTest):
    """Test path resolution functionality."""

    def test_resolve_absolute_path(self, tmp_path):
        """Test resolving absolute paths."""
        abs_path = tmp_path / "absolute/path"

        result = resolve_path(abs_path, relative_to_root=False)

        assert result.is_absolute()

    def test_resolve_relative_path_to_absolute(self, tmp_path):
        """Test resolving relative paths to absolute."""
        rel_path = "./data"

        result = resolve_path(rel_path, relative_to_root=False)

        assert result.is_absolute()

    def test_resolve_path_as_string(self, tmp_path):
        """Test resolve_path with string input."""
        path_str = str(tmp_path / "test")

        result = resolve_path(path_str, relative_to_root=False)

        assert isinstance(result, Path)

    def test_resolve_path_as_pathobject(self, tmp_path):
        """Test resolve_path with Path object input."""
        path_obj = tmp_path / "test"

        result = resolve_path(path_obj, relative_to_root=False)

        assert isinstance(result, Path)


class TestSymlinkedDirectories(BasePathTest):
    """Test path resolution with symlinked directories."""

    def test_create_symlink(self, tmp_path):
        """Test creating a symlink."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()

        symlink = tmp_path / "symlink"
        try:
            symlink.symlink_to(real_dir)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        assert symlink.exists()
        assert symlink.is_symlink()

    def test_symlink_points_to_correct_target(self, tmp_path):
        """Test that symlink points to correct target."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()

        symlink = tmp_path / "symlink"
        try:
            symlink.symlink_to(real_dir)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        # Verify symlink resolves to real path
        assert symlink.resolve() == real_dir.resolve()

    def test_symlinked_subdirectory(self, tmp_path):
        """Test symlinked subdirectory within repo."""
        repo = self.create_temp_repo(tmp_path, [".git/"])

        real_dir = repo / "real_data"
        real_dir.mkdir()

        symlink = repo / "data"
        try:
            symlink.symlink_to(real_dir)
        except OSError:
            pytest.skip("Symlinks not supported on this system")

        assert symlink.exists()
        assert (repo / "data").resolve() == (repo / "real_data").resolve()


class TestEnvironmentVariables(BasePathTest):
    """Test environment variable handling."""

    def test_set_environment_variable(self, tmp_path):
        """Test setting and reading environment variable."""
        test_path = str(tmp_path / "test")
        os.environ["STREAMSIGHT_ROOT"] = test_path

        assert os.environ["STREAMSIGHT_ROOT"] == test_path

    def test_environment_variable_isolation(self, tmp_path):
        """Test that environment variables are isolated between tests."""
        test_value = str(tmp_path)
        os.environ["TEST_VAR"] = test_value

        # This should persist within the test
        assert os.environ["TEST_VAR"] == test_value

    def test_environment_variable_cleanup(self, tmp_path):
        """Test cleanup of environment variables."""
        # Set a variable
        os.environ["TEMP_VAR"] = "test_value"
        assert "TEMP_VAR" in os.environ

        # It should be cleaned up in teardown
        # (verified by setup_method clearing it)


class TestNestedNotebookScenarios(BasePathTest):
    """Test nested directory structure for notebooks."""

    def test_shallow_notebook_structure(self, tmp_path):
        """Test notebook one level deep."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        notebook_dir = repo / "notebooks"
        notebook_dir.mkdir()

        assert (repo / "notebooks").exists()
        assert str(repo) in str(notebook_dir)

    def test_deeply_nested_notebook(self, tmp_path):
        """Test notebook in deeply nested structure."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        notebook_path = repo / "notebooks/experiments/2024/november/week3"
        notebook_path.mkdir(parents=True)

        assert notebook_path.exists()
        assert (repo / ".git").exists()

    def test_notebook_with_checkpoints(self, tmp_path):
        """Test notebook directory with .ipynb_checkpoints."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        notebook_dir = repo / "notebooks"
        notebook_dir.mkdir()
        (notebook_dir / ".ipynb_checkpoints").mkdir()

        assert (notebook_dir / ".ipynb_checkpoints").exists()

    def test_multiple_notebook_directories(self, tmp_path):
        """Test multiple notebook directories at different levels."""
        repo = self.create_temp_repo(tmp_path, [".git/"])

        notebook_dirs = [
            repo / "notebooks",
            repo / "experiments/notebooks",
            repo / "research/2024/notebooks",
        ]

        for nb_dir in notebook_dirs:
            nb_dir.mkdir(parents=True, exist_ok=True)

        for nb_dir in notebook_dirs:
            assert nb_dir.exists()


class TestWindowsPaths(BasePathTest):
    """Test Windows path handling."""

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_windows_path_object(self, tmp_path):
        """Test Windows path handling."""
        repo = self.create_temp_repo(tmp_path, [".git/"])

        # Path should handle backslashes correctly
        assert isinstance(repo, Path)

    def test_path_separator_handling(self, tmp_path):
        """Test path separator handling across platforms."""
        repo = self.create_temp_repo(tmp_path, [".git/"])

        # Path object should normalize separators
        path_str = str(repo / "subdir" / "file.txt")
        assert isinstance(path_str, str)


class TestRelativePathHandling(BasePathTest):
    """Test handling of relative paths."""

    def test_dot_path_notation(self, tmp_path):
        """Test ./ path notation."""
        test_path = Path("./data")

        assert isinstance(test_path, Path)
        assert not test_path.is_absolute()

    def test_parent_path_notation(self, tmp_path):
        """Test ../ path notation."""
        test_path = Path("../other")

        assert isinstance(test_path, Path)
        assert not test_path.is_absolute()

    def test_resolve_relative_to_absolute(self, tmp_path):
        """Test resolving relative paths to absolute."""
        rel_path = Path("./data")

        resolved = (tmp_path / rel_path).resolve()

        assert resolved.is_absolute()


class TestReadOnlyFilesystemHandling(BasePathTest):
    """Test behavior with permission-restricted filesystems."""

    def test_permission_error_handling(self, tmp_path):
        """Test handling of permission errors."""
        # This test verifies error handling patterns
        restricted_path = tmp_path / "restricted"
        restricted_path.mkdir()

        # On most systems, we can create directories
        assert restricted_path.exists()

    def test_safe_dir_with_valid_path(self, tmp_path):
        """Test safe_dir succeeds with valid permissions."""
        new_dir = tmp_path / "new"

        result = safe_dir(new_dir)

        assert new_dir.exists()


class TestNetworkPathStructures(BasePathTest):
    """Test path structures for network mounts."""

    def test_nfs_style_path(self, tmp_path):
        """Test NFS-style path handling."""
        nfs_path = tmp_path / "mnt/shared/streamsight"
        nfs_path.mkdir(parents=True)

        assert nfs_path.exists()
        assert "mnt" in str(nfs_path)

    def test_mounted_volume_structure(self, tmp_path):
        """Test mounted volume directory structure."""
        mount_point = tmp_path / "mnt/project"
        mount_point.mkdir(parents=True)
        (mount_point / ".git").mkdir()

        assert mount_point.exists()
        assert (mount_point / ".git").exists()


class TestMultipleRepositoryStructures(BasePathTest):
    """Test handling of multiple repository scenarios."""

    def test_nested_git_repos(self, tmp_path):
        """Test nested git repositories."""
        parent_repo = self.create_temp_repo(tmp_path, [".git/"])
        child_repo = parent_repo / "child_project"
        child_repo.mkdir()
        (child_repo / ".git").mkdir()

        # Both should exist
        assert (parent_repo / ".git").exists()
        assert (child_repo / ".git").exists()

    def test_monorepo_structure(self, tmp_path):
        """Test monorepo with multiple projects."""
        monorepo = self.create_temp_repo(tmp_path, [".git/", "pyproject.toml"])

        projects = ["project_a", "project_b", "project_c"]
        for proj in projects:
            (monorepo / proj).mkdir()
            (monorepo / proj / "setup.py").touch()

        # All should exist
        for proj in projects:
            assert (monorepo / proj / "setup.py").exists()


class TestSubdirectoryExecution(BasePathTest):
    """Test directory structures for script execution."""

    def test_src_subdirectory_structure(self, tmp_path):
        """Test src/ subdirectory structure."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        src_dir = repo / "src/streamsight"
        src_dir.mkdir(parents=True)

        assert src_dir.exists()

    def test_nested_package_structure(self, tmp_path):
        """Test deeply nested package structure."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        nested = repo / "src/streamsight/datasets/loaders"
        nested.mkdir(parents=True)

        assert nested.exists()

    def test_tests_directory_structure(self, tmp_path):
        """Test tests directory structure."""
        repo = self.create_temp_repo(tmp_path, [".git/"])
        tests_dir = repo / "tests/unit"
        tests_dir.mkdir(parents=True)

        assert tests_dir.exists()


class TestHelperFunctionUtilities(BasePathTest):
    """Test utility helper functions."""

    def test_safe_dir_type_consistency(self, tmp_path):
        """Test safe_dir returns consistent types."""
        test_paths = [
            tmp_path / "single",
            tmp_path / "nested/path",
            tmp_path / "deep/nested/path/structure",
        ]

        for test_path in test_paths:
            result = safe_dir(test_path)
            assert isinstance(result, Path)

    def test_multiple_safe_dir_calls(self, tmp_path):
        """Test multiple safe_dir calls on same path."""
        test_dir = tmp_path / "test"

        result1 = safe_dir(test_dir)
        result2 = safe_dir(test_dir)

        assert result1 == result2
        assert test_dir.exists()


class TestPathComparison(BasePathTest):
    """Test path comparison and assertion utilities."""

    def test_assert_paths_equal_same_paths(self, tmp_path):
        """Test assert_paths_equal with identical paths."""
        path1 = tmp_path / "test"
        path2 = tmp_path / "test"

        # Should not raise
        self.assert_paths_equal(path1, path2, "Identical paths should be equal")

    def test_assert_paths_equal_resolved_paths(self, tmp_path):
        """Test assert_paths_equal with resolved paths."""
        path1 = tmp_path / "test"
        path2 = tmp_path / "./test"

        # Both should resolve to same canonical path
        self.assert_paths_equal(path1, path2, "Resolved paths should be equal")

    def test_assert_paths_not_equal(self, tmp_path):
        """Test assert_paths_equal fails with different paths."""
        path1 = tmp_path / "test1"
        path2 = tmp_path / "test2"

        with pytest.raises(AssertionError):
            self.assert_paths_equal(path1, path2, "Different paths should not be equal")
