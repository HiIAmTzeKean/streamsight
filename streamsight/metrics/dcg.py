import logging
import itertools
from typing import Optional

from scipy.sparse import csr_matrix
import numpy as np
from streamsight.metrics.base import ListwiseMetricK
from streamsight.metrics.util import sparse_divide_nonzero


logger = logging.getLogger(__name__)


class DCGK(ListwiseMetricK):
    """Computes the sum of gains of all items in a recommendation list.

    Relevant items that are ranked higher in the Top-K recommendations have a higher gain.

    The Discounted Cumulative Gain (DCG) is computed for every user as

    .. math::

        \\text{DiscountedCumulativeGain}(u) = \\sum\\limits_{i \\in Top-K(u)} \\frac{y^{true}_{u,i}}{\\log_2 (\\text{rank}(u,i) + 1)}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K=10, timestamp_limit: Optional[int] = None):
        super().__init__(K,timestamp_limit)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        denominator = y_pred_top_K.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self._scores = csr_matrix(dcg.sum(axis=1))

        return