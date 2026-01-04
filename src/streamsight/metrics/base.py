import logging
from abc import abstractmethod
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix

from ..algorithms.utils import get_top_K_ranks
from ..models import BaseModel, ParamMixin


logger = logging.getLogger(__name__)


class Metric(BaseModel, ParamMixin):
    """Base class for all metrics.

    A Metric object is stateful, i.e. after `calculate`
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`
    """

    _scores: None | csr_matrix
    _user_id_map: np.ndarray
    _y_true: csr_matrix
    _y_pred: csr_matrix
    _true_positive: int
    """Number of true positives computed. Used for caching to obtain macro results."""
    _false_negative: int
    """Number of false negatives computed. Used for caching to obtain macro results."""
    _false_positive: int
    """Number of false positives computed. Used for caching to obtain macro results."""

    def __init__(
        self,
        timestamp_limit: None | int = None,
    ) -> None:
        self._num_users: int = 0
        self._num_items: int = 0
        self._timestamp_limit: None | int = timestamp_limit
        self._is_time_aware: bool = timestamp_limit is not None

    @property
    def _is_computed(self) -> bool:
        """Whether the metric has been computed."""
        return hasattr(self, "_scores")

    def get_params(self) -> dict[str, int | None]:
        """Get the parameters of the metric."""
        if not self._is_time_aware:
            return {}
        return {"timestamp_limit": self._timestamp_limit}

    @property
    def identifier(self) -> str:
        """Name of the metric."""
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items() if v is not None))
        return f"{self.__class__.__name__}({paramstring})"

    @abstractmethod
    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        raise NotImplementedError()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Calculates this metric for all nonzero users in `y_true`,
        given true labels and predicted scores.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        y_true, y_pred = self._eliminate_empty_users(y_true=y_true, y_pred=y_pred)
        if not self.is_y_true_pred_shape_match(y_true=y_true, y_pred=y_pred):
            raise AssertionError(
                f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}"
            )
        self._set_shape(y_true)
        self._calculate(y_true, y_pred)

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        """Micro results for the metric.

        :return: Detailed results for the metric.
        :rtype: dict[str, np.ndarray]
        """
        return {"score": np.array(self.macro_result)}

    @property
    def macro_result(self) -> float:
        """The global metric value."""
        raise NotImplementedError()

    @property
    def timestamp_limit(self) -> int:
        """The timestamp limit for the metric."""
        if not self._is_time_aware or self._timestamp_limit is None:
            raise ValueError("This metric is not time-aware.")
        return self._timestamp_limit

    @property
    def num_items(self) -> int:
        """Dimension of the item-space in both `y_true` and `y_pred`"""
        return self._num_items

    @property
    def num_users(self) -> int:
        """Dimension of the user-space in both `y_true` and `y_pred`
        after elimination of users without interactions in `y_true`.
        """
        return self._num_users

    @property
    def _indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Indices in the prediction matrix for which scores were computed."""
        row, col = np.indices((self._num_users, self._num_items))

        return row.flatten(), col.flatten()

    def is_y_true_pred_shape_match(self, y_true: csr_matrix, y_pred: csr_matrix) -> bool:
        """Make sure the dimensions of y_true and y_pred match."""
        return y_true.shape == y_pred.shape

    def _set_shape(self, y_true: csr_matrix) -> None:
        """Set the number of users and items in the metric.

        The values of `self._num_users` and `self._num_items` are set
        to the number of users and items in `y_true`. This allows for the
        computation of the metric to be done in the correct shape.

        :param y_true: Binary representation of user-item interactions.
        :type y_true: csr_matrix
        """
        self._num_users, self._num_items = y_true.shape

    def _eliminate_empty_users(
        self, y_true: csr_matrix, y_pred: csr_matrix
    ) -> tuple[csr_matrix, csr_matrix]:
        """Eliminate users that have no interactions in `y_true`.

        Users with no interactions in `y_true` are eliminated from the
        prediction matrix `y_pred`. This is done to avoid division by zero
        and to also reduce the computational overhead.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: (y_true, y_pred), with zero users eliminated.
        :rtype: tuple[csr_matrix, csr_matrix]
        """
        # Get the rows (users) that are not empty
        nonzero_users = list(set(y_true.nonzero()[0]))

        self._user_id_map = np.array(nonzero_users)

        return y_true[nonzero_users, :], y_pred[nonzero_users, :]

    def _map_users(self, users):
        """Map internal identifiers of users to actual user identifiers."""
        if hasattr(self, "_user_id_map"):
            return self._user_id_map[users]
        else:
            return users


class MetricTopK(Metric):
    """Base class for all metrics computed based on the Top-K recommendations for every user.

    A MetricTopK object is stateful, i.e. after `calculate`
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    DEFAULT_K = 10

    def __init__(
        self,
        K: None | int = DEFAULT_K,
        timestamp_limit: None | int = None,
    ) -> None:
        super().__init__(timestamp_limit)
        if K is None:
            warn(f"K not specified, using default value {self.DEFAULT_K}.")
            K = self.DEFAULT_K
        self.K = K

    @property
    def name(self) -> str:
        """Name of the metric."""
        return f"{super().name}_{self.K}"

    @property
    def params(self) -> dict[str, int | None]:
        """Parameters of the metric."""
        return super().params | {"K": self.K}

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row, col = self.y_pred.nonzero()
        return row, col

    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Computes metric given true labels `y_true` and predicted scores `y_pred`. Only Top-K recommendations are considered.

        To be implemented in the child class.

        :param y_true: Expected interactions per user.
        :type y_true: csr_matrix
        :param y_pred: Ranks for topK recommendations per user
        :type y_pred: csr_matrix
        """
        raise NotImplementedError()

    def prepare_matrix(
        self, y_true: csr_matrix, y_pred: csr_matrix
    ) -> tuple[csr_matrix, csr_matrix]:
        """Prepare the matrices for the metric calculation.

        This method is used to prepare the matrices for the metric calculation.
        It is used to eliminate empty users and to set the shape of the matrices.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: tuple of the prepared matrices.
        :rtype: tuple[csr_matrix, csr_matrix]
        """
        # Perform checks and cleaning
        y_true, y_pred = self._eliminate_empty_users(y_true=y_true, y_pred=y_pred)
        if not self.is_y_true_pred_shape_match(y_true, y_pred):
            raise AssertionError(
                f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}"
            )
        self._set_shape(y_true=y_true)

        # Compute the topK for the predicted affinities
        y_pred = get_top_K_ranks(y_pred, self.K)

        return y_true, y_pred

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Computes metric given true labels `y_true` and predicted scores `y_pred`. Only Top-K recommendations are considered.

        Detailed metric results can be retrieved with :attr:`results`.
        Global aggregate metric value is retrieved as :attr:`value`.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        # Perform checks and cleaning
        # TODO check if y_true is empty?
        y_true, y_pred = self.prepare_matrix(y_true, y_pred)
        self.y_pred = y_pred

        self._calculate(y_true, y_pred)


class ListwiseMetricK(MetricTopK):
    """Base class for all listwise metrics that can be calculated for every Top-K recommendation list,
    i.e. one value for each user.
    Examples are: PrecisionK, RecallK, DCGK, NDCGK.

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def col_names(self) -> list[str]:
        """The names of the columns in the results DataFrame."""
        return ["user_id", "score"]

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row = np.arange(self.y_pred.shape[0])
        col = np.zeros(self.y_pred.shape[0], dtype=np.int32)
        return row, col

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        """User level results for the metric.

        Contains an entry for every user.

        :return: The results DataFrame with columns: user_id, score
        :rtype: pd.DataFrame
        """
        if not self._is_computed:
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            logger.warning(UserWarning("No scores were computed. Returning empty dict."))
            return dict(zip(self.col_names, (np.array([]), np.array([]))))

        scores = self._scores.toarray()

        int_users, items = self._indices
        values = scores[int_users, items]

        users = self._map_users(int_users)

        return dict(zip(self.col_names, (users, values)))

    @property
    def macro_result(self) -> None | float:
        """Global metric value obtained by taking the average over all users.

        :raises ValueError: If the metric has not been calculated yet.
        :return: The global metric value.
        :rtype: float, optional
        """
        if not self._is_computed:
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            logger.warning(UserWarning("No scores were computed. Returning Null value."))
            return None
        elif self._scores.size == 0:
            logger.warning(
                UserWarning(
                    f"All predictions were off or the ground truth matrix was empty during compute of {self.identifier}."
                )
            )
            return 0
        return self._scores.mean()


class ElementwiseMetricK(MetricTopK):
    """Base class for all elementwise metrics that can be calculated for
    each user-item pair in the Top-K recommendations.

    :attr:`results` contains an entry for each user-item pair.

    Examples are: HitK

    This code is adapted from RecPack :cite:`recpack`

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def col_names(self) -> list[str]:
        """The names of the columns in the results DataFrame."""
        return ["user_id", "item_id", "score"]

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        if not self._is_computed:
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            warn(UserWarning("No scores were computed. Returning empty dict."))
            return dict(zip(self.col_names, (np.array([]), np.array([]))))

        scores = self._scores.toarray()

        all_users = set(range(self._scores.shape[0]))
        int_users, items = self._indices
        values = scores[int_users, items]

        # For all users in all_users but not in int_users,
        # add K items np.nan with value = 0
        missing_users = all_users.difference(set(int_users))

        # This should barely occur, so it's not too bad to append np arrays.
        for u in list(missing_users):
            for i in range(self.K):
                int_users = np.append(int_users, u)
                values = np.append(values, 0)
                items = np.append(items, np.nan)

        users = self._map_users(int_users)

        return dict(zip(self.col_names, (users, items, values)))

    @property
    def macro_result(self) -> None | float:
        if not self._is_computed:
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            logger.warning(UserWarning("No scores were computed. Returning Null value."))
            return None
        elif self._scores.size == 0:
            logger.warning(
                UserWarning(
                    f"All predictions were off or the ground truth matrix was empty during compute of {self.identifier}."
                )
            )
            return 0

        hit_ratio = self._scores.sum(axis=1) / self.y_pred.shape[1]
        return hit_ratio.mean()
        # return self._scores.sum(axis=1).mean()
