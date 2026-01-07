
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.utils import get_top_K_values
from streamsight.matrix import ItemUserBasedEnum
from streamsight.utils.util import add_rows_to_csr_matrix

logger = logging.getLogger(__name__)


def compute_cosine_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the cosine similarity between the items in the matrix.

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :return: similarity matrix
    :rtype: csr_matrix
    """
    # X.T otherwise we are doing a user KNN
    item_cosine_similarities = cosine_similarity(X.T, dense_output=False)
    item_cosine_similarities.setdiag(0)
    # Set diagonal to 0, because we don't want to support self similarity

    return item_cosine_similarities


class ItemKNN(Algorithm):
    """Item K Nearest Neighbours model.

    First described in 'Item-based top-n recommendation algorithms.' :cite:`10.1145/963770.963776`
    
    This code is adapted from RecPack :cite:`recpack`

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.

    Cosine similarity between item i and j is computed as

    .. math::
        sim(i,j) = \\frac{X_i X_j}{||X_i||_2 ||X_j||_2}

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    """
    ITEM_USER_BASED = ItemUserBasedEnum.ITEM
    
    def __init__(
        self,
        K=10
    ):
        super().__init__()
        self.K = K

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item
        We assume that X is a binary matrix of shape (n_users, n_items)
        """
        item_similarities = compute_cosine_similarity(X)
        item_similarities = get_top_K_values(item_similarities, K=self.K)

        self.similarity_matrix_ = item_similarities

    def _predict(self, X: csr_matrix, predict_frame:Optional[pd.DataFrame]=None) -> csr_matrix:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :param predict_frame: DataFrame containing the user IDs to predict for
        :type predict_frame: pd.DataFrame, optional
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        scores = X @ self.similarity_matrix_
        return scores

    def _pad_predict(self, X_pred: csr_matrix, intended_shape: tuple, to_predict_frame: pd.DataFrame) -> csr_matrix:
        """Pad the predictions with random items for users that are not in the training data.

        :param X_pred: Predictions made by the algorithm
        :type X_pred: csr_matrix
        :param intended_shape: The intended shape of the prediction matrix
        :type intended_shape: tuple
        :param to_predict_frame: DataFrame containing the user IDs to predict for
        :type to_predict_frame: pd.DataFrame
        :return: The padded prediction matrix
        :rtype: csr_matrix
        """
        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0]-known_user_id)
        # pad users with random items
        logger.debug(f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with random items")
        to_predict = to_predict_frame.value_counts("uid")
        row = []
        col = []
        for user_id in to_predict.index:
            if user_id >= known_user_id:
                # For top-K algo, the request could be to predict only 1 item for a user
                # but by definition we should return top-K items for each user
                row += [user_id] * self.K
                col += self.rand_gen.integers(0, known_item_id, self.K).tolist()
        pad = csr_matrix((np.ones(len(row)), (row, col)), shape=intended_shape)
        X_pred += pad
        logger.debug(f"Padding completed")
        return X_pred