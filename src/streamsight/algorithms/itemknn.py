import logging
from typing import Self

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from streamsight.matrix import ItemUserBasedEnum, PredictionMatrix
from .base import PopularityPaddingMixin, TopKItemSimilarityMatrixAlgorithm
from .utils import get_top_K_values


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
    if not isinstance(item_cosine_similarities, csr_matrix):
        item_cosine_similarities = csr_matrix(item_cosine_similarities)
    # Set diagonal to 0, because we don't want to support self similarity
    item_cosine_similarities.setdiag(0)

    return item_cosine_similarities


class ItemKNN(TopKItemSimilarityMatrixAlgorithm, PopularityPaddingMixin):
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

    def _fit(self, X: csr_matrix) -> Self:
        """Fit a cosine similarity matrix from item to item
        We assume that X is a binary matrix of shape (n_users, n_items)
        """
        item_similarities = compute_cosine_similarity(X)
        item_similarities = get_top_K_values(item_similarities, K=self.K)

        self.similarity_matrix_ = item_similarities
        self.X_ = X.copy()
        return self

    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        predict_ui_df = X.get_prediction_data()._df  # noqa: SLF001

        # create a boolean series that is true for index in predict_ui_df.uid
        uid_to_predict = predict_ui_df[predict_ui_df.uid < self.X_.shape[0]].uid.unique()
        uid_to_predict = sorted(uid_to_predict.tolist())

        # features: csr_matrix = self.X_[uid_to_predict]
        # we try without any filtering on the feature matrix
        features: csr_matrix = self.X_
        scores = features @ self.similarity_matrix_

        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        intended_shape = (X.max_global_user_id, X.max_global_item_id)

        if scores.shape == intended_shape:
            return scores

        # there are 2 cases where the shape is different:
        # 1. The algorithm did not predict unknown user, causing shortage in rows
        # 2. The algorithm not aware of unknown items, causing shortage in columns

        # handle case 1
        if scores.shape[1] < intended_shape[1]:
            scores = self._pad_unknown_iid_with_none_strategy(
                y_pred=scores,
                current_shape=scores.shape,
                intended_shape=intended_shape,
            )

        # handle case 2
        if self.pad_with_popularity:
            scores = self._pad_uknown_uid_with_popularity_strategy(
                X_pred=scores,
                intended_shape=intended_shape,
                predict_ui_df=predict_ui_df,
            )
        else:
            # current_shape = (X.max_known_user_id, X.max_known_item_id)
            scores = self._pad_unknown_uid_with_random_strategy(
                X_pred=scores,
                current_shape=scores.shape,
                # current_shape=current_shape,
                intended_shape=intended_shape,
                predict_ui_df=predict_ui_df,
            )

        pred = scores[predict_ui_df["uid"].values]
        return pred
