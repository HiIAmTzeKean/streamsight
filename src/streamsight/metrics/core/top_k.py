import logging
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix

from ...algorithms.utils import get_top_K_ranks
from .base import Metric


logger = logging.getLogger(__name__)


class MetricTopK(Metric):
    """Base class for all metrics computed based on the Top-K recommendations for every user.

    A MetricTopK object is stateful, i.e. after `calculate`
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(
        self,
        user_id_sequence_array: np.ndarray,
        user_item_shape: tuple[int, int],
        timestamp_limit: None | int = None,
        K: int = 10,
    ) -> None:
        super().__init__(
            user_id_sequence_array=user_id_sequence_array,
            user_item_shape=user_item_shape,
            timestamp_limit=timestamp_limit,
        )
        if K is None:
            warn(f"K not specified, using default value {K}.")
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
    def col_names(self) -> list[str]:
        """The names of the columns in the results DataFrame."""
        return ["user_id", "score"]

    def prepare_matrix(self, y_true: csr_matrix, y_pred: csr_matrix) -> tuple[csr_matrix, csr_matrix]:
        y_true, y_pred = super()._prepare_matrix(y_true, y_pred)
        y_pred = get_top_K_ranks(y_pred, self.K)
        return y_true, y_pred
