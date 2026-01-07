import logging

import scipy.sparse
from scipy.sparse import csr_matrix

from streamsight.metrics.base import ListwiseMetricK

logger = logging.getLogger(__name__)


class PrecisionK(ListwiseMetricK):
    """Computes the fraction of top-K recommendations that correspond
    to true interactions.

    Given the prediction and true interaction in binary representation,
    the matrix is multiplied elementwise. These will result in the true
    positives to be 1 and the false positives to be 0. The sum of the
    resulting true positives is then divided by the number of actual top-K
    interactions to get the precision on user level.
    
    In simple terms, precision is the ratio of correctly predicted positive
    observations to the total predictions made.

    Precision is computed per user as:

    .. math::

        \\text{Precision}(u) = \\frac{\\sum\\limits_{i \\in \\text{Top-K}(u)} y^{true}_{u,i}}{K}\\
    
    ref: RecPack
    
    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # obtain true positives
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1
        scores = scores.tocsr()

        # true positive/total predictions
        self._scores = csr_matrix(scores.sum(axis=1)) / self.K

        logger.debug(f"Precision compute complete - {self.name}")