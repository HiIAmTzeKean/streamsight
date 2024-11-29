"""Path utilities for streamsight library.

This module provides functions to resolve paths relative to the repository root,
ensuring that data and logs are stored consistently regardless of where the
library is run from.
"""

import os
from pathlib import Path
from typing import Optional


_REPO_ROOT_CACHE: Optional[Path] = None


def get_repo_root(
    marker_files: tuple[str, ...] = (
        ".git",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "README.md",
        "requirements.txt",
    ),
) -> Path:
    """Find and return the repository root directory.

    The function searches upward from the current file's directory looking for
    common repository marker files (for example, `.git` or `pyproject.toml`).
    The result is cached to avoid repeated filesystem traversal.

    Args:
        marker_files: Tuple of filenames that indicate the repository root.

    Returns:
        Path to the repository root directory.

    Raises:
        RuntimeError: If the repository root cannot be located and the
            `STREAMSIGHT_ROOT` environment variable is not set or invalid.
    """
    global _REPO_ROOT_CACHE

    if _REPO_ROOT_CACHE is not None:
        return _REPO_ROOT_CACHE

    # Start from this file's location
    current = Path(__file__).resolve().parent
    max_depth = 10

    for _ in range(max_depth):
        # Check for marker files
        if any((current / marker).exists() for marker in marker_files):
            _REPO_ROOT_CACHE = current
            return current

        # Move up one directory
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # Fallback: try environment variable
    if "STREAMSIGHT_ROOT" in os.environ:
        root = Path(os.environ["STREAMSIGHT_ROOT"])
        if root.exists():
            _REPO_ROOT_CACHE = root
            return root

    raise RuntimeError(
        "Could not find repository root. Please set STREAMSIGHT_ROOT "
        "environment variable or ensure you're running from within the repo."
    )


def get_data_dir(subdir: str = "") -> Path:
    """Return the `data/` directory inside the repository.

    Args:
        subdir: Optional subdirectory within `data/` to append.

    Returns:
        Path to the data directory (with `subdir` appended when provided).
    """
    data_dir = get_repo_root() / "data"
    if subdir:
        data_dir = data_dir / subdir
    return data_dir


def get_logs_dir(subdir: str = "") -> Path:
    """Return the `logs/` directory inside the repository.

    Args:
        subdir: Optional subdirectory within `logs/` to append.

    Returns:
        Path to the logs directory (with `subdir` appended when provided).
    """
    logs_dir = get_repo_root() / "logs"
    if subdir:
        logs_dir = logs_dir / subdir
    return logs_dir


def get_cache_dir(subdir: str = "") -> Path:
    """Return the `cache/` directory inside the repository.

    Args:
        subdir: Optional subdirectory within `cache/` to append.

    Returns:
        Path to the cache directory (with `subdir` appended when provided).
    """
    cache_dir = get_repo_root() / "cache"
    if subdir:
        cache_dir = cache_dir / subdir
    return cache_dir


def safe_dir(path: Path | str) -> Path:
    """Ensure the given directory exists, creating it if necessary.

    Args:
        path: Directory path as a :class:`pathlib.Path` or string.

    Returns:
        The directory path (created if it did not exist).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path, relative_to_root: bool = True) -> Path:
    """Resolve a path to an absolute :class:`pathlib.Path`.

    Args:
        path: Path to resolve, either a string or :class:`pathlib.Path`.
        relative_to_root: If True and `path` is relative, resolve it relative to
            the repository root. If False, resolve it relative to the current
            working directory.

    Returns:
        The resolved absolute path.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if relative_to_root:
        return get_repo_root() / path

    return path.resolve()
