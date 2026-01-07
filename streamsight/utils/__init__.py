"""
.. currentmodule:: streamsight.utils 

Utility
-------------
The utility module provides a set of utility functions that are used in the
Streamsight library. These function are general functions that are not class
specific and also contains functions that are used in the top level such as
preparing the logger and creating the configuration file.

.. autosummary::
    :toctree: generated/

    create_config_yaml
    safe_dir
    add_columns_to_csr_matrix
    add_rows_to_csr_matrix
    arg_to_str
    df_to_sparse
    prepare_logger
    to_binary
    to_tuple
    ProgressBar
"""

from streamsight.utils.directory_tools import create_config_yaml, safe_dir
from streamsight.utils.util import (
    add_columns_to_csr_matrix,
    add_rows_to_csr_matrix,
    arg_to_str,
    df_to_sparse,
    prepare_logger,
    to_binary,
    to_tuple,
    ProgressBar
)
