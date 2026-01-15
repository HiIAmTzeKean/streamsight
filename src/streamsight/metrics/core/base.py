import logging
from abc import abstractmethod
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix

from ...algorithms.utils import get_top_K_ranks
from ...models import BaseModel, ParamMixin


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
    _user_id_sequence_array: np.ndarray
    """Sequence of user IDs in the evaluation data."""
    _num_users: int
    _true_positive: int
    """Number of true positives computed. Used for caching to obtain macro results."""
    _false_negative: int
    """Number of false negatives computed. Used for caching to obtain macro results."""
    _false_positive: int
    """Number of false positives computed. Used for caching to obtain macro results."""

    def __init__(
        self,
        user_id_sequence_array: np.ndarray,
        user_item_shape: tuple[int, int],
        timestamp_limit: None | int = None,
    ) -> None:
        self._user_id_sequence_array = user_id_sequence_array
        self._num_users, self._num_items = user_item_shape
        self._timestamp_limit: None | int = timestamp_limit

    @property
    def _is_computed(self) -> bool:
        """Whether the metric has been computed."""
        return hasattr(self, "_scores")

    def get_params(self) -> dict[str, int | None]:
        """Get the parameters of the metric."""
        if not self.is_time_aware:
            return {}
        return {"timestamp_limit": self._timestamp_limit}

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        """Micro results for the metric.

        :return: Detailed results for the metric.
        :rtype: dict[str, np.ndarray]
        """
        return {"score": np.array(self.macro_result)}

    @property
    def macro_result(self) -> None | float:
        """The global metric value."""
        raise NotImplementedError()

    @property
    def is_time_aware(self) -> bool:
        """Whether the metric is time-aware."""
        return self._timestamp_limit is not None

    @property
    def timestamp_limit(self) -> int:
        """The timestamp limit for the metric."""
        if not self.is_time_aware or self._timestamp_limit is None:
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

    def _prepare_matrix(
        self, y_true: csr_matrix, y_pred: csr_matrix
    ) -> tuple[csr_matrix, csr_matrix]:
        """Prepare the matrices for the metric calculation.

        This method is used to prepare the matrices for the metric calculation.
        It is used to eliminate empty users and to set the shape of the matrices.
        """
        if not y_true.shape == y_pred.shape:
            raise AssertionError(
                f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}"
            )
        self._set_shape(y_true=y_true)
        return y_true, y_pred

    @abstractmethod
    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        raise NotImplementedError()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Calculates this metric for all nonzero users in `y_true`,
        given true labels and predicted scores.
        """
        y_true, y_pred = self._prepare_matrix(y_true, y_pred)
        self._calculate(y_true, y_pred)

    def _set_shape(self, y_true: csr_matrix) -> None:
        """Set the number of users and items based on the shape of y_true.
        """
        self._num_users, self._num_items = y_true.shape  # type: ignore
