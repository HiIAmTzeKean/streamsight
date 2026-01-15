# Adopted from RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging

import numpy as np
from scipy.sparse import csr_matrix

from ..core.listwise_top_k import ListwiseMetricK
from ..core.util import sparse_divide_nonzero


logger = logging.getLogger(__name__)


class DCGK(ListwiseMetricK):
    """Computes the sum of gains of all items in a recommendation list.

    Relevant items that are ranked higher in the Top-K recommendations have a higher gain.

    The Discounted Cumulative Gain (DCG) is computed for every user as

    .. math::

        \\text{DiscountedCumulativeGain}(u) = \\sum\\limits_{i \\in Top-K(u)} \\frac{y^{true}_{u,i}}{\\log_2 (\\text{rank}(u,i) + 1)}


    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int

    This code is adapted from RecPack :cite:`recpack`
    """
    IS_BASE: bool = False

    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        logger.debug("Precision compute started - %s", self.name)
        logger.debug("Shape of matrix: (%d, %d)", y_true.shape[0], y_true.shape[1])
        logger.debug("Number of ground truth interactions: %d", y_true.nnz)

        denominator = y_pred.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        self._scores = csr_matrix(dcg.sum(axis=1))

        logger.debug(f"DCGK compute complete - {self.name}")
