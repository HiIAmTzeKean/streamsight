"""Utility module for Streamsight library.

This module provides a set of general utility functions used throughout Streamsight.
It includes utilities for file handling, configuration, matrix operations, and logging.

## Utility Functions

General-purpose utility functions that support library operations:

- `create_config_yaml`: Create configuration YAML file
- `safe_dir`: Safely manage directory operations
- `add_columns_to_csr_matrix`: Add columns to sparse matrix
- `add_rows_to_csr_matrix`: Add rows to sparse matrix
- `arg_to_str`: Convert arguments to string representation
- `df_to_sparse`: Convert DataFrame to sparse matrix
- `to_binary`: Convert data to binary format
- `to_tuple`: Convert data to tuple format
- `ProgressBar`: Progress bar utility for tracking operations

## Path Utilities

Directory and path management functions:

- `get_cache_dir`: Get cache directory path
- `get_data_dir`: Get data directory path
- `get_logs_dir`: Get logs directory path
- `get_repo_root`: Get repository root directory
- `safe_dir`: Safely create and manage directories

## Logging Control

Functions to control logging level and warning suppression:

- `log_level`: Get current logging level
- `log_level_by_name`: Set logging level by name (DEBUG, INFO, WARNING, ERROR)
- `prepare_logger`: Initialize logger for Streamsight
- `suppress_warnings`: Suppress all Python warnings
- `suppress_specific_warnings`: Suppress specific warning types

## Logging Example

```python
import logging
import warnings
import streamsight

# Set log level to INFO
streamsight.log_level_by_name("INFO")

# Suppress all warnings
streamsight.suppress_warnings(suppress=True)

# Log information
logger = logging.getLogger("streamsight")
logger.info("This is an informational message.")

# Warnings will be suppressed
warnings.warn("This warning will not appear.")
```

## Configuration

- `create_config_yaml`: Generate configuration YAML file for Streamsight
"""

from streamsight.utils.logging_tools import (
    log_level,
    log_level_by_name,
    prepare_logger,
    suppress_specific_warnings,
    suppress_warnings,
)
from streamsight.utils.path import (
    get_cache_dir,
    get_data_dir,
    get_logs_dir,
    get_repo_root,
    safe_dir,
)
from streamsight.utils.util import (
    ProgressBar,
    add_columns_to_csr_matrix,
    add_rows_to_csr_matrix,
    arg_to_str,
    df_to_sparse,
    to_binary,
    to_tuple,
)
from streamsight.utils.yaml_tool import create_config_yaml


__all__ = [
    "create_config_yaml",
    "safe_dir",
    "add_columns_to_csr_matrix",
    "add_rows_to_csr_matrix",
    "arg_to_str",
    "df_to_sparse",
    "prepare_logger",
    "to_binary",
    "to_tuple",
    "ProgressBar",
    "log_level",
    "log_level_by_name",
    "suppress_warnings",
    "suppress_specific_warnings",
    "get_cache_dir",
    "get_data_dir",
    "get_logs_dir",
    "get_repo_root",
]
