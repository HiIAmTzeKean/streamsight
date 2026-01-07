from typing import Union

from scipy.sparse import csr_matrix

from streamsight.utils.util import to_binary

from streamsight.matrix.interaction_matrix import InteractionMatrix

Matrix = Union[InteractionMatrix, csr_matrix]

def to_csr_matrix(
    X: Matrix, binary: bool = False
) -> csr_matrix:
    """Convert a matrix-like object to a scipy csr_matrix.

    :param X: Matrix-like object or tuple of objects to convert.
    :type X: csr_matrix
    :param binary: If true, ensure matrix is binary by setting non-zero values to 1.
    :type binary: bool, optional
    :return: Matrices as csr_matrix.
    :rtype: Union[csr_matrix, Tuple[csr_matrix, ...]]
    """
    
    if isinstance(X, csr_matrix):
        res = X
    elif isinstance(X, InteractionMatrix):
        res = X.values
    else:
        raise AttributeError("Not supported Matrix conversion")
    return to_binary(res) if binary else res