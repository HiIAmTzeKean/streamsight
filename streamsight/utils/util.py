import logging
import logging.config
import os
from typing import Union

import numpy as np
import progressbar
import yaml
from scipy.sparse import csr_matrix

from streamsight.utils.directory_tools import create_config_yaml, safe_dir

logger = logging.getLogger(__name__)


def to_tuple(el):
    """Whether single element or tuple, always returns as tuple."""
    if type(el) == tuple:
        return el
    else:
        return (el,)


def arg_to_str(arg: Union[type, str]) -> str:
    """Converts a type to its name or returns the string.
    
    :param arg: Argument to convert to string.
    :type arg: Union[type, str]
    :return: String representation of the argument.
    :rtype: str
    :raises TypeError: If the argument is not a string or a type.
    """
    if type(arg) == type:
        arg = arg.__name__

    elif type(arg) != str:
        raise TypeError(f"Argument should be string or type, not {type(arg)}!")

    return arg


def df_to_sparse(df, item_ix, user_ix, value_ix=None, shape=None):
    if value_ix is not None and value_ix in df:
        values = df[value_ix]
    else:
        if value_ix is not None:
            # value_ix provided, but not in df
            logger.warning(
                f"Value column {value_ix} not found in dataframe. Using ones instead."
            )

        num_entries = df.shape[0]
        # Scipy sums up the entries when an index-pair occurs more than once,
        # resulting in the actual counts being stored. Neat!
        values = np.ones(num_entries)

    indices = list(zip(*df.loc[:, [user_ix, item_ix]].values))

    if indices == []:
        indices = [[], []]  # Empty zip does not evaluate right

    if shape is None:
        shape = df[user_ix].max() + 1, df[item_ix].max() + 1
    sparse_matrix = csr_matrix(
        (values, indices), shape=shape, dtype=values.dtype
    )

    return sparse_matrix


def to_binary(X: csr_matrix) -> csr_matrix:
    """Converts a matrix to binary by setting all non-zero values to 1.

    :param X: Matrix to convert to binary.
    :type X: csr_matrix
    :return: Binary matrix.
    :rtype: csr_matrix
    """
    X_binary = X.astype(bool).astype(X.dtype)

    return X_binary


def invert(x: Union[np.ndarray, csr_matrix]) -> Union[np.ndarray, csr_matrix]:
    """Invert an array.

    :param x: [description]
    :type x: [type]
    :return: [description]
    :rtype: [type]
    """
    if isinstance(x, np.ndarray):
        ret = np.zeros(x.shape)
    elif isinstance(x, csr_matrix):
        ret = csr_matrix(x.shape)
    else:
        raise TypeError("Unsupported type for argument x.")
    ret[x.nonzero()] = 1 / x[x.nonzero()]
    return ret


class ProgressBar:
    """Progress bar as visual.
    """
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def prepare_logger(path: str) -> dict:
    """Prepare the logger.
    
    Prepare the logger by reading the configuration file and setting up the logger.
    If the configuration file does not exist, it will be created.
    
    :param path: Path to the configuration file.
    :type path: str
    :return: Configuration dictionary.
    :rtype: dict
    """
    if not os.path.exists(path):
        create_config_yaml(path)

    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    dir_name = os.path.dirname(config["handlers"]["file"]["filename"])
    safe_dir(dir_name)

    logging.config.dictConfig(config)
    logging.captureWarnings(True)
    return config


def add_rows_to_csr_matrix(matrix: csr_matrix, n: int = 1) -> csr_matrix:
    """Add a row of zeros to a csr_matrix.

    ref: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr

    :param matrix: Matrix to add a row of zeros to.
    :type matrix: csr_matrix
    :return: Matrix with a row of zeros added.
    :rtype: csr_matrix
    """
    new_shape = (matrix.shape[0] + n, matrix.shape[1])
    new_indptr = np.append(matrix.indptr, [matrix.indptr[-1]] * n)
    matrix = csr_matrix(
        (matrix.data, matrix.indices, new_indptr), shape=new_shape, copy=False
    )
    return matrix


def add_columns_to_csr_matrix(matrix: csr_matrix, n: int = 1) -> csr_matrix:
    """Add a column of zeros to a csr_matrix.

    https://stackoverflow.com/questions/30691160/effectively-change-dimension-of-scipy-spare-csr-matrix

    :param matrix: Matrix to add a column of zeros to.
    :type matrix: csr_matrix
    :return: Matrix with a column of zeros added.
    :rtype: csr_matrix
    """
    new_shape = (matrix.shape[0], matrix.shape[1] + n)
    matrix = csr_matrix(
        (matrix.data, matrix.indices, matrix.indptr),
        shape=new_shape,
        copy=False,
    )
    return matrix


def set_row_csr_addition(
    A: csr_matrix, row_idx: int, new_row: np.ndarray
) -> None:
    """Set row of a csr_matrix to a new row.

    ref: https://stackoverflow.com/questions/28427236/set-row-of-csr-matrix

    :param A: Matrix to set a row of.
    :type A: csr_matrix
    :param row_idx: Index of the row to set.
    :type row_idx: int
    :param new_row: New row to set.
    :type new_row: np.ndarray
    """
    indptr = np.zeros(A.shape[1] + 1)
    indptr[row_idx + 1 :] = A.shape[1]
    indices = np.arange(A.shape[1])
    A += csr_matrix((new_row, indices, indptr), shape=A.shape)
