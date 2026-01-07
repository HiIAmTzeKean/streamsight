import logging
from typing import Dict, Optional, Tuple
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, vstack
from deprecation import deprecated

from streamsight.algorithms.utils import get_top_K_ranks
from streamsight.utils.util import add_columns_to_csr_matrix


logger = logging.getLogger(__name__)


class Metric:
    """Base class for all metrics.

    A Metric object is stateful, i.e. after ``calculate``
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`
    """

    def __init__(self,
                 timestamp_limit: Optional[int] = None,
                 cache: bool = False):
        self._num_users = 0
        self._num_items = 0
        self._timestamp_limit = timestamp_limit
        self.cache = cache

        self._scores: Optional[csr_matrix]
        self._value: float
        self._y_true: csr_matrix
        self._y_pred: csr_matrix
        self._true_positive: int
        """Number of true positives computed. Used for caching to obtain macro results."""
        self._false_negative: int
        """Number of false negatives computed. Used for caching to obtain macro results."""
        self._false_positive: int
        """Number of false positives computed. Used for caching to obtain macro results."""

    @property
    def name(self):
        """Name of the metric."""
        return self.__class__.__name__

    @property
    def params(self):
        """Parameters of the metric."""
        return {"timestamp_limit": self._timestamp_limit}

    def get_params(self):
        """Get the parameters of the metric."""
        return self.params

    @property
    def identifier(self):
        """Name of the metric."""
        # return f"{super().identifier[:-1]},K={self.K})"
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items() if v is not None))
        return self.__class__.__name__ + "(" + paramstring + ")"

    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        raise NotImplementedError()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Calculates this metric for all nonzero users in ``y_true``,
        given true labels and predicted scores.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        y_true, y_pred = self._eliminate_empty_users(y_true, y_pred)
        self._verify_shape(y_true, y_pred)
        self._set_shape(y_true)
        self._calculate(y_true, y_pred)

    @deprecated(
        details="Caching values for metric is no longer needed for core functionalities due to change in compute method."
    )
    def cache_values(self, y_true:csr_matrix, y_pred:csr_matrix) -> None:
        """Cache the values of y_true and y_pred for later use.
        
        Basic method to cache the values of y_true and y_pred for later use.
        This is useful when the metric can be calculated with the cumulative
        values of y_true and y_pred.
        
        .. note::
            This method should be over written in the child class if the metric
            cannot be calculated with the cumulative values of y_true and y_pred.
            For example, in the case of Precision@K, the metric default behavior
            is to obtain the top-K ranks of y_pred and and y_true, this will
            cause cumulative values to be possibly dropped.
            
        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :raises ValueError: If caching is disabled for the metric.
        """
        if not self.cache:
            raise ValueError("Caching is disabled for this metric.")

        if not hasattr(self, "_y_true") or not hasattr(self, "_y_pred"):
            self._y_true = y_true
            self._y_pred = y_pred
            return

        # reshape old y_true and y_pred to add the new columns
        if y_true.shape[1] > self._y_true.shape[1]:
            self._y_true = add_columns_to_csr_matrix(self._y_true, y_true.shape[1]-self._y_true.shape[1])
            self._y_pred = add_columns_to_csr_matrix(self._y_pred, y_pred.shape[1]-self._y_pred.shape[1])

        # ? np.vstack([self._y_true.toarray(), y_true.toarray()]) faster ?
        self._y_true = vstack([self._y_true, y_true])
        self._y_pred = vstack([self._y_pred, y_pred])

    @deprecated(
        details="Caching values for metric is no longer needed for core functionalities due to change in compute method."
    )
    def calculate_cached(self):
        """Calculate the metric using the cached values of y_true and y_pred.
        
        This method calculates the metric using the cached values of y_true and y_pred.
        :meth:`calculate` will be called on the cached values.
        
        .. note::
            This method should be overwritten in the child class if the metric
            cannot be calculated with the cumulative values of y_true and y_pred.

        :raises ValueError: If caching is disabled for the metric.
        """
        if not self.cache:
            raise ValueError("Caching is disabled for this metric.")
        if not hasattr(self, "_y_true") or not hasattr(self, "_y_pred"):
            self._scores = None
            return

        self.calculate(self._y_true, self._y_pred)

    @property
    def micro_result(self) -> Dict[str, np.ndarray]:
        """Micro results for the metric.
        
        :return: Detailed results for the metric.
        :rtype: Dict[str, np.ndarray]
        """
        return {"score": np.array(self.macro_result)}

    @property
    def macro_result(self) -> float:
        """The global metric value."""
        if not hasattr(self, "_value"):
            raise ValueError("Metric has not been calculated yet.")
        return self._value

    @property
    def timestamp_limit(self):
        """The timestamp limit for the metric."""
        return self._timestamp_limit

    @property
    def num_items(self) -> int:
        """Dimension of the item-space in both ``y_true`` and ``y_pred``"""
        return self._num_items

    @property
    def num_users(self) -> int:
        """Dimension of the user-space in both ``y_true`` and ``y_pred``
        after elimination of users without interactions in ``y_true``.
        """
        return self._num_users

    @property
    def _indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Indices in the prediction matrix for which scores were computed."""
        row, col = np.indices((self._num_users, self._num_items))

        return row.flatten(), col.flatten()

    def _verify_shape(self, y_true: csr_matrix, y_pred: csr_matrix) -> bool:
        """Make sure the dimensions of y_true and y_pred match.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :raises AssertionError: Shape mismatch between y_true and y_pred.
        :return: True if dimensions match.
        :rtype: bool
        """
        check = y_true.shape == y_pred.shape
        if not check:
            raise AssertionError(f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}")
        return check

    def _set_shape(self, y_true: csr_matrix) -> None:
        """Set the number of users and items in the metric.
        
        The values of ``self._num_users`` and ``self._num_items`` are set
        to the number of users and items in ``y_true``. This allows for the
        computation of the metric to be done in the correct shape.

        :param y_true: Binary representation of user-item interactions.
        :type y_true: csr_matrix
        """
        self._num_users, self._num_items = y_true.shape

    def _eliminate_empty_users(self, y_true: csr_matrix, y_pred: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """Eliminate users that have no interactions in ``y_true``.

        Users with no interactions in ``y_true`` are eliminated from the
        prediction matrix ``y_pred``. This is done to avoid division by zero
        and to also reduce the computational overhead.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: (y_true, y_pred), with zero users eliminated.
        :rtype: Tuple[csr_matrix, csr_matrix]
        """
        # Get the rows (users) that are not empty
        nonzero_users = list(set(y_true.nonzero()[0]))

        self.user_id_map_ = np.array(nonzero_users)

        return y_true[nonzero_users, :], y_pred[nonzero_users, :]

    def _map_users(self, users):
        """Map internal identifiers of users to actual user identifiers."""
        if hasattr(self, "user_id_map_"):
            return self.user_id_map_[users]
        else:
            return users


class MetricTopK(Metric):
    """Base class for all metrics computed based on the Top-K recommendations for every user.

    A MetricTopK object is stateful, i.e. after ``calculate``
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """
    DEFAULT_K = 10

    def __init__(self, K:Optional[int] = DEFAULT_K,
                 timestamp_limit: Optional[int] = None,
                 cache: bool = False):
        super().__init__(timestamp_limit, cache)
        if K is None:
            warn(f"K not specified, using default value {self.DEFAULT_K}.")
            K = self.DEFAULT_K
        self.K = K

    @property
    def name(self):
        """Name of the metric."""
        return f"{super().name}_{self.K}"
    
    @property
    def params(self):
        """Parameters of the metric."""
        return super().params | {"K": self.K}

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row, col = self.y_pred_top_K_.nonzero()
        return row, col

    def _calculate(self, y_true:csr_matrix, y_pred_top_K:csr_matrix):
        """Computes metric given true labels ``y_true`` and predicted scores ``y_pred``. Only Top-K recommendations are considered.

        To be implemented in the child class.

        :param y_true: Expected interactions per user.
        :type y_true: csr_matrix
        :param y_pred_top_K: Ranks for topK recommendations per user
        :type y_pred_top_K: csr_matrix
        """
        raise NotImplementedError()
    
    def prepare_matrix(self, y_true: csr_matrix, y_pred: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """Prepare the matrices for the metric calculation.

        This method is used to prepare the matrices for the metric calculation.
        It is used to eliminate empty users and to set the shape of the matrices.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: Tuple of the prepared matrices.
        :rtype: Tuple[csr_matrix, csr_matrix]
        """
        # Perform checks and cleaning
        y_true, y_pred = self._eliminate_empty_users(y_true, y_pred)
        self._verify_shape(y_true, y_pred)
        self._set_shape(y_true)

        # Compute the topK for the predicted affinities
        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        
        return y_true, y_pred_top_K
    
    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Computes metric given true labels ``y_true`` and predicted scores ``y_pred``. Only Top-K recommendations are considered.

        Detailed metric results can be retrieved with :attr:`results`.
        Global aggregate metric value is retrieved as :attr:`value`.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        # Perform checks and cleaning
        # TODO check if y_true is empty?
        y_true, y_pred_top_K = self.prepare_matrix(y_true, y_pred)
        self.y_pred_top_K_ = y_pred_top_K
        
        self._calculate(y_true, y_pred_top_K)


class ListwiseMetricK(MetricTopK):
    """Base class for all metrics that can be calculated for every Top-K recommendation list,
    i.e. one value for each user.
    Examples are: PrecisionK, RecallK

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def col_names(self):
        """The names of the columns in the results DataFrame."""
        return ["user_id", "score"]

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row = np.arange(self.y_pred_top_K_.shape[0])
        col = np.zeros(self.y_pred_top_K_.shape[0], dtype=np.int32)
        return row, col

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        """User level results for the metric.

        Contains an entry for every user.

        :return: The results DataFrame with columns: user_id, score
        :rtype: pd.DataFrame
        """
        if not hasattr(self, "_scores"):
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            warn(UserWarning("No scores were computed. Returning empty dict."))
            return dict(zip(self.col_names, (np.array([]), np.array([]))))
    
        scores = self._scores.toarray()

        int_users, items = self._indices
        values = scores[int_users, items]

        users = self._map_users(int_users)

        return dict(zip(self.col_names, (users, values)))
    

    @property
    def macro_result(self) -> Optional[float]:
        """Global metric value obtained by taking the average over all users.

        :raises ValueError: If the metric has not been calculated yet.
        :return: The global metric value.
        :rtype: float, optional
        """
        if not hasattr(self, "_scores"):
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            warn(UserWarning("No scores were computed. Returning Null value."))
            return None
        elif self._scores.size == 0:
            warn(UserWarning(f"All predictions were off or the ground truth matrix was empty during compute of {self.identifier}."))
            return 0
        return self._scores.mean()
