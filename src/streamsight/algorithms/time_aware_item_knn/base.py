# Adopted from RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from warnings import warn

import numpy as np
from recpack.algorithms.nearest_neighbour import (
    compute_conditional_probability,
    compute_cosine_similarity,
    compute_pearson_similarity,
)
from recpack.algorithms.time_aware_item_knn.decay_functions import (
    ConcaveDecay,
    ConvexDecay,
    ExponentialDecay,
    InverseDecay,
    LinearDecay,
    LogDecay,
    NoDecay,
)
from recpack.util import get_top_K_values
from scipy.sparse import csr_matrix

from streamsight.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from streamsight.matrix import InteractionMatrix, Matrix
from streamsight.utils.util import add_rows_to_csr_matrix


EPSILON = 1e-13

logger = logging.getLogger(__name__)


class TARSItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Framework for time aware variants of the ItemKNN algorithm.

    This class was inspired by works from Liu, Nathan N., et al. (2010), Ding et al. (2005) and Lee et al. (2007).

    The framework for these approaches can be summarised as:

    - When training the user interaction matrix is weighted to take into account temporal information.
    - Similarities are computed on this weighted matrix, using various similarity measures.
    - When predicting the interactions are similarly weighted, giving more weight to more recent interactions.
    - Recommendation scores are obtained by multiplying the weighted interaction matrix with
      the previously computed similarity matrix.

    The similarity between items is based on their decayed interaction vectors:

    .. math::

        \\text{sim}(i,j) = s(\\Gamma(A_i), \\Gamma(A_j))

    Where :math:`s` is a similarity function (like ``cosine``),
    :math:`\\Gamma` a decay function (like ``exponential_decay``) and
    :math:`A_i` contains the distances to now from when the users interacted with item `i`,
    if they interacted with the item at all (else the value is 0).

    During computation, 'now' is considered as the maximal timestamp in the matrix + 1.
    As such the age is always a positive non-zero value.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, Optional
    :param pad_with_popularity: Whether to pad the similarity matrix with RecentPop Algorithm.
        Defaults to True.
    :type pad_with_popularity: bool, optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to `` 1 / (24 * 3600)`` (one day).
    :type fit_decay: float, optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to ``1 / (24 * 3600)`` (one day).
    :type predict_decay: float, optional
    :param decay_interval: Size of a single time unit in seconds.
        Allows more finegrained parameters for large scale datasets where events are collected over months of data.
        Defaults to 1 (second).
    :type decay_interval: int, optional
    :param similarity: Which similarity measure to use. Defaults to ``"cosine"``.
        ``["cosine", "conditional_probability", "pearson"]`` are supported.
    :type similarity: str, Optional
    :param decay_function: The decay function to use, defaults to ``"exponential"``.
        Supported values are ``["exponential", "log", "linear", "concave", "convex", "inverse"]``

    This code is adapted from RecPack :cite:`recpack`
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability", "pearson"]
    DECAY_FUNCTIONS = {
        "exponential": ExponentialDecay,
        "log": LogDecay,
        "linear": LinearDecay,
        "concave": ConcaveDecay,
        "convex": ConvexDecay,
        "inverse": InverseDecay,
    }

    def __init__(
        self,
        K: int = 200,
        pad_with_popularity: bool = True,
        fit_decay: float = 1 / (24 * 3600),
        predict_decay: float = 1 / (24 * 3600),
        decay_interval: int = 1,
        similarity: str = "cosine",
        decay_function: str = "exponential",
    ):
        # Uses other default parameters for ItemKNN
        super().__init__(K=K)
        self.training_data: InteractionMatrix = None
        self.pad_with_popularity = pad_with_popularity

        if decay_interval <= 0 or type(decay_interval) == float:
            raise ValueError("Parameter decay_interval needs to be a positive integer.")

        self.decay_interval = decay_interval

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"Similarity {similarity} is not supported.")
        self.similarity = similarity

        if decay_function not in self.DECAY_FUNCTIONS:
            raise ValueError(f"Decay function {decay_function} is not supported.")

        self.decay_function = decay_function

        # Verify decay parameters
        if self.decay_function in ["exponential", "log", "linear", "concave", "convex"]:
            if fit_decay != 0:
                self.DECAY_FUNCTIONS[decay_function].validate_decay(fit_decay)

            if predict_decay != 0:
                self.DECAY_FUNCTIONS[decay_function].validate_decay(predict_decay)

        self.fit_decay = fit_decay
        self.predict_decay = predict_decay
        self.decay_function = decay_function

    def _get_decay_func(self, decay, max_value):
        if decay == 0:
            return NoDecay()

        elif self.decay_function == "inverse":
            return self.DECAY_FUNCTIONS[self.decay_function]()
        elif self.decay_function in ["exponential", "convex"]:
            return self.DECAY_FUNCTIONS[self.decay_function](decay)
        elif self.decay_function in ["log", "linear", "concave"]:
            return self.DECAY_FUNCTIONS[self.decay_function](decay, max_value)

    def _predict(self, X: csr_matrix, predict_im: InteractionMatrix) -> csr_matrix:
        """Predict scores for nonzero users in X.

        Scores are computed by matrix multiplication of weighted X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        X_decay = self._add_decay_to_predict_matrix(self.training_data)
        X_pred = super()._predict(X_decay)

        # ID indexing starts at 0, so max_id + 1 is the number of unique IDs
        max_user_id = predict_im.max_user_id + 1
        max_item_id = predict_im.max_item_id + 1
        intended_shape = (
            max(max_user_id, X.shape[0]),
            max(max_item_id, X.shape[1]),
        )

        predict_frame = predict_im._df

        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0] - known_user_id)
        logger.debug(f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with items")
        to_predict = predict_frame.value_counts("uid")

        if self.pad_with_popularity:
            popular_items = self.get_popularity_scores(super()._transform_fit_input(X))
            for user_id in to_predict.index:
                if user_id >= known_user_id:
                    X_pred[user_id, :] = popular_items
        else:
            row = []
            col = []
            for user_id in to_predict.index:
                if user_id >= known_user_id:
                    row += [user_id] * to_predict[user_id]
                    col += self.rand_gen.integers(0, known_item_id, to_predict[user_id]).tolist()
            pad = csr_matrix((np.ones(len(row)), (row, col)), shape=intended_shape)
            X_pred += pad

        logger.debug(f"Padding by {self.name} completed")
        return X_pred

    def get_popularity_scores(self, X: csr_matrix):
        """Pad the predictions with popular items for users that are not in the training data."""
        interaction_counts = X.sum(axis=0).A[0]
        sorted_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if num_items < self.K:
            warn("K is larger than the number of items.", UserWarning)

        K = min(self.K, num_items)
        ind = np.argpartition(sorted_scores, -K)[-K:]
        a = np.zeros(X.shape[1])
        a[ind] = sorted_scores[ind]

        return a

    def _transform_fit_input(self, X: Matrix) -> InteractionMatrix:
        """Weigh each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix) -> InteractionMatrix:
        """Weigh each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: csr_matrix) -> "TARSItemKNN":
        """Fit a cosine similarity matrix from item to item."""

        if self.training_data is None:
            self.training_data = X.copy()
        else:
            self.training_data = self.training_data.union(X)
        X = self.training_data.copy()

        X = self._add_decay_to_fit_matrix(X)
        if self.similarity == "cosine":
            item_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X)
        elif self.similarity == "pearson":
            item_similarities = compute_pearson_similarity(X)

        item_similarities = get_top_K_values(item_similarities, K=self.K)

        self.similarity_matrix_ = item_similarities

        return self

    def _add_decay_to_interaction_matrix(self, X: InteractionMatrix, decay: float) -> csr_matrix:
        """Weigh the interaction matrix based on age of the events.

        If decay is 0, it is assumed to be disabled, and so we just return binary matrix.
        :param X: Interaction matrix.
        :type X: InteractionMatrix
        :return: Weighted csr matrix.
        :rtype: csr_matrix
        """
        timestamp_mat = X.latest_interaction_timestamps_matrix

        # To get 'now', we add 1 to the maximal timestamp. This makes sure there are no vanishing zeroes.
        now = timestamp_mat.data.max() + 1
        ages = (now - timestamp_mat.data) / self.decay_interval
        timestamp_mat.data = self._get_decay_func(decay, ages.max())(ages)

        return csr_matrix(timestamp_mat)

    def _add_decay_to_fit_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X, self.fit_decay)

    def _add_decay_to_predict_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X, self.predict_decay)
