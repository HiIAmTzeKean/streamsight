import logging

import scipy.sparse
from scipy.sparse import csr_matrix

from ..core.listwise_top_k import ListwiseMetricK


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
    IS_BASE: bool = False

    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        scores = scipy.sparse.lil_matrix(y_pred.shape)

        logger.debug("Precision compute started - %s", self.name)
        logger.debug("Shape of matrix: (%d, %d)", y_true.shape[0], y_true.shape[1])
        logger.debug("Number of ground truth interactions: %d", y_true.nnz)

        # obtain true positives
        scores[y_pred.multiply(y_true).astype(bool)] = 1
        scores = scores.tocsr()

        # true positive/total predictions
        self._scores = csr_matrix(scores.sum(axis=1)) / self.K
        logger.debug("Precision compute complete - %s", self.name)
