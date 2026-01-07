import logging

import scipy.sparse
from scipy.sparse import csr_matrix

from streamsight.metrics.base import ListwiseMetricK


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

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # obtain true positives
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1
        scores = scores.tocsr()

        # true positive/total actual interactions
        self._scores = csr_matrix(scores.sum(axis=1) / y_true.sum(axis=1))

        logger.debug(f"Recall compute complete - {self.name}")
