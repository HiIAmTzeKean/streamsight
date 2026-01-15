import logging

import scipy.sparse
from scipy.sparse import csr_matrix

from ..core.listwise_top_k import ListwiseMetricK


logger = logging.getLogger(__name__)


class RecallK(ListwiseMetricK):
    """Computes the fraction of true interactions that made it into
    the Top-K recommendations.

    Recall per user is computed as:

    .. math::

        \\text{Recall}(u) = \\frac{\\sum\\limits_{i \\in \\text{Top-K}(u)} y^{true}_{u,i} }{\\sum\\limits_{j \\in I} y^{true}_{u,j}}

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

        # true positive/total actual interactions
        self._scores = csr_matrix(scores.sum(axis=1) / y_true.sum(axis=1))
        logger.debug(f"Recall compute complete - {self.name}")
