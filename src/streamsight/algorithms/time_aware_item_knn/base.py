# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.metrics.pairwise import cosine_similarity

from streamsight.algorithms.base import TopKItemSimilarityMatrixAlgorithm
from streamsight.matrix import InteractionMatrix, Matrix
from streamsight.utils.util import add_rows_to_csr_matrix


EPSILON = 1e-13

logger = logging.getLogger(__name__)


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


def to_binary(X: csr_matrix) -> csr_matrix:
    """Converts a matrix to binary by setting all non-zero values to 1.

    :param X: Matrix to convert to binary.
    :type X: csr_matrix
    :return: Binary matrix.
    :rtype: csr_matrix
    """
    X_binary = X.astype(bool).astype(X.dtype)

    return X_binary


class DecayFunction:
    def __call__(self, time_distances: ArrayLike) -> ArrayLike:
        """Apply the decay.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: Array of event ages to which decays have been applied.
        :rtype: ArrayLike
        """
        raise NotImplementedError()


class ExponentialDecay(DecayFunction):
    """Applies exponential decay.

    For each value x in ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = e^{-\\alpha * x}

    where alpha is the decay parameter.

    :param decay: Exponential decay parameter, should be in the [0, 1] interval.
    :type decay: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (0 <= decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: [0, 1].")

    def __init__(self, decay: float):
        self.validate_decay(decay)
        self.decay = decay

    def __call__(self, time_distances: ArrayLike) -> ArrayLike:
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """

        return np.exp(-self.decay * time_distances)


class ConvexDecay(DecayFunction):
    """Applies a convex decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = \\alpha^{x}

    where :math:`alpha` is the decay parameter.

    :param decay: The decay parameter, should be in the ]0, 1] interval.
    :type decay: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (0 < decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]0, 1].")

    def __init__(self, decay: float):
        self.validate_decay(decay)
        self.decay = decay

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """

        return np.power(self.decay, time_distances)


class ConcaveDecay(DecayFunction):
    """Applies a concave decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = 1 - \\alpha^{1-\\frac{x}{N}}

    where :math:`alpha` is the decay parameter and :math:`N` is the ``max_distance`` parameter.

    :param decay: The decay parameter, should be in the [0, 1[ interval.
    :type decay: float
    :param max_distance: Normalizing parameter, to put distances in the [0, 1].
    :type max_distance: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (0 < decay <= 1):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]0, 1].")

    def __init__(self, decay: float, max_distance: float):
        self.validate_decay(decay)
        self.decay = decay
        self.max_distance = max_distance

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed array.
        :rtype: ArrayLike
        """
        if (time_distances > self.max_distance).any():
            raise ValueError(
                "At least one of the distances is bigger than the specified max_distance."
            )
        return 1 - np.power(self.decay, 1 - (time_distances / self.max_distance))


class LogDecay(DecayFunction):
    """Applies a logarithmic decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = log_\\alpha ((\\alpha-1)(1-\\frac{x}{N}) + 1)

    where :math:`alpha` is the decay parameter and :math:`N` is the ``max_distance`` parameter.

    :param decay: The decay parameter, should be in the range ]1, inf[
    :type decay: float
    :param max_distance: Normalizing parameter, to put distances in the [0, 1].
    :type max_distance: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        """Verify if the decay parameter is in the right range for this decay function."""
        if not (1 < decay):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: ]1, inf[.")

    def __init__(self, decay: float, max_distance: float):
        self.validate_decay(decay)
        self.decay = decay
        self.max_distance = max_distance

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """
        if (time_distances > self.max_distance).any():
            raise ValueError(
                "At least one of the distances is bigger than the specified max_distance."
            )
        return np.log(((self.decay - 1) * (1 - time_distances / self.max_distance)) + 1) / np.log(
            self.decay
        )


class LinearDecay(DecayFunction):
    """Applies a linear decay function.

    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = \\max(1 - (\\frac{x}{N}) \\alpha, 0)

    where :math:`alpha` is the decay parameter and :math:`N` is the ``max_distance`` parameter.

    :param decay: The decay parameter, should be in the [0, inf[ interval.
    :type decay: float
    :param max_distance: Normalizing parameter, to put distances in the [0, 1].
    :type max_distance: float
    """

    @classmethod
    def validate_decay(cls, decay: float):
        if not (0 <= decay):
            raise ValueError(f"Decay parameter = {decay} is not in the supported range: [0, +inf[.")

    def __init__(self, decay: float, max_distance: float):
        self.validate_decay(decay)
        self.decay = decay
        self.max_distance = max_distance

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """
        if (time_distances > self.max_distance).any():
            raise ValueError(
                "At least one of the distances is bigger than the specified max_distance."
            )
        results = 1 - (time_distances / self.max_distance) * self.decay
        results[results < 0] = 0
        return results


class InverseDecay(DecayFunction):
    """Invert the scores.

    Decay parameter only added for interface unity.
    For each value x in the ``time_distances`` the decayed value is computed as

    .. math::

        f(x) = \\frac{1}{x}
    """

    def __call__(self, time_distances: ArrayLike):
        """Apply the decay function.

        :param time_distances: array of distances to be decayed.
        :type time_distances: ArrayLike
        :returns: The decayed time array.
        :rtype: ArrayLike
        """

        results = time_distances.astype(float).copy()
        results[results > 0] = 1 / results[results > 0]
        results[results == 0] = 1
        return results


class NoDecay(ExponentialDecay):
    """Turns the array into a binary array."""

    def __init__(self):
        super().__init__(0)


def compute_conditional_probability(X: csr_matrix, pop_discount: float = 0) -> csr_matrix:
    """Compute conditional probability like similarity.

    Computation using equation (3) from the original ItemKNN paper.
    'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis

    .. math ::
        sim(i,j) = \\frac{\\sum\\limits_{u \\in U} \\mathbb{I}_{u,i} X_{u,j}}{Freq(i) \\times Freq(j)^{\\alpha}}

    Where :math:`\\mathbb{I}_{ui}` is 1 if the user u has visited item i, and 0 otherwise.
    And alpha is the pop_discount parameter.
    Note that this is a non-symmetric similarity measure.
    Given that X is a binary matrix, and alpha is set to 0,
    this simplifies to pure conditional probability.

    .. math::
        sim(i,j) = \\frac{Freq(i \\land j)}{Freq(i)}

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :param pop_discount: Parameter defining popularity discount. Defaults to 0
    :type pop_discount: float, Optional.
    """
    # matrix with co_mat_i,j =  SUM(1_u,i * X_u,j for each user u)
    # If the input matrix is binary, this is the cooccurence count matrix.
    co_mat = to_binary(X).T @ X

    # Compute the inverse of the item frequencies
    A = invert(diags(to_binary(X).sum(axis=0).A[0]).tocsr())

    if pop_discount:
        # This has all item similarities
        # Co_mat is weighted by both the frequencies of item i
        # and the frequency of item j to the pop_discount power.
        # If pop_discount = 1, this similarity is symmetric again.
        item_cond_prob_similarities = A @ co_mat @ A.power(pop_discount)
    else:
        # Weight the co_mat with the amount of occurences of item i.
        item_cond_prob_similarities = A @ co_mat

    # Set diagonal to 0, because we don't support self similarity
    item_cond_prob_similarities.setdiag(0)

    return item_cond_prob_similarities


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


def compute_pearson_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the pearson correlation as a similarity between each item in the matrix.

    Self similarity is removed.
    When computing similarity, the avg of nonzero entries per user is used.

    :param X: Rating or psuedo rating matrix.
    :type X: csr_matrix
    :return: similarity matrix.
    :rtype: csr_matrix
    """

    if (X == 1).sum() == X.nnz:
        raise ValueError("Pearson similarity can not be computed on a binary matrix.")

    count_per_item = (X > 0).sum(axis=0).A

    avg_per_item = X.sum(axis=0).A.astype(float)

    avg_per_item[count_per_item > 0] = (
        avg_per_item[count_per_item > 0] / count_per_item[count_per_item > 0]
    )

    X = X - (X > 0).multiply(avg_per_item)

    # Given the rescaled matrix, the pearson correlation is just cosine similarity on this matrix.
    return compute_cosine_similarity(X)


def get_top_K_ranks(X: csr_matrix, K: Optional[int] = None) -> csr_matrix:
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
            top_k_row = X.indices[
                le + np.argpartition(X.data[le:ri], list(range(-K_row_pick, 0)))[-K_row_pick:]
            ]

            for rank, col_ix in enumerate(reversed(top_k_row)):
                U.append(row_ix)
                I.append(col_ix)
                V.append(rank + 1)

    X_top_K = csr_matrix((V, (U, I)), shape=X.shape)

    return X_top_K


def get_top_K_values(X: csr_matrix, K: Optional[int] = None) -> csr_matrix:
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
    top_K_ranks[top_K_ranks > 0] = 1  # ranks to binary

    return top_K_ranks.multiply(X)  # elementwise multiplication


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
