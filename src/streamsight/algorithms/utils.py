import numpy as np
from scipy.sparse import csr_matrix


def get_top_K_ranks(X: csr_matrix, K: None | int = None) -> csr_matrix:
    """Returns a matrix of ranks assigned to the largest K values in X.

    Selects K largest values for every row in X and assigns a rank to each.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :param K: Amount of values to select.
    :type K: int, optional
    :return: Matrix with K values per row.
    :rtype: csr_matrix
    """
    U, I, V = [], [], []
    for row_ix, (le, ri) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
        K_row_pick = min(K, ri - le) if K is not None else ri - le

        if K_row_pick != 0:
            top_k_row = X.indices[le + np.argpartition(X.data[le:ri], list(range(-K_row_pick, 0)))[-K_row_pick:]]

            for rank, col_ix in enumerate(reversed(top_k_row)):
                U.append(row_ix)
                I.append(col_ix)
                V.append(rank + 1)
    # data, (row, col) = (V, (U, I)
    X_top_K = csr_matrix((V, (U, I)), shape=X.shape)

    return X_top_K


def get_top_K_values(X: csr_matrix, K: None | int = None) -> csr_matrix:
    """Returns a matrix of only the K largest values for every row in X.

    Selects the top-K items for every user (which is equal to the K nearest neighbours.)
    In case of a tie for the last position, the item with the largest index of the tied items is used.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :param K: Amount of values to select.
    :type K: int, optional
    :return: Matrix with K values per row.
    :rtype: csr_matrix
    """
    top_K_ranks = get_top_K_ranks(X, K)
    # Convert the position into binary values (1 if in top K, 0 otherwise)
    top_K_ranks[top_K_ranks > 0] = 1
    # elementwise multiplication with orignal matrix to get values
    return top_K_ranks.multiply(X)
