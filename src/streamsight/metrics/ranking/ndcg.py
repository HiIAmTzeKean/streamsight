# Adopted from RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..core.listwise_top_k import ListwiseMetricK
from ..core.util import sparse_divide_nonzero


logger = logging.getLogger(__name__)


class NDCGK(ListwiseMetricK):

    """Computes the normalized sum of gains of all items in a recommendation list.

    The normalized Discounted Cumulative Gain (nDCG) is similar to DCG,
    but normalizes by dividing the resulting sum of cumulative gains
    by the best possible discounted cumulative gain for a list of recommendations
    of length K for a user with history length N.

    Scores are always in the interval [0, 1]

    .. math::

        \\text{NormalizedDiscountedCumulativeGain}(u) = \\frac{\\text{DCG}(u)}{\\text{IDCG}(u)}

    where IDCG stands for Ideal Discounted Cumulative Gain, computed as:

    .. math::

        \\text{IDCG}(u) = \\sum\\limits_{j=1}^{\\text{min}(K, |y^{true}_u|)} \\frac{1}{\\log_2 (j + 1)}

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int

    This code is adapted from RecPack :cite:`recpack`
    """
    IS_BASE: bool = False

    def _calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        logger.debug(f"NDCGK compute started - {self.name}")
        logger.debug(f"Number of users: {y_true.shape[0]}")
        logger.debug(f"Number of ground truth interactions: {y_true.nnz}")

        self.discount_template = 1.0 / np.log2(np.arange(2, self.K + 2))
        # Calculate IDCG values by creating a list of partial sums
        self.IDCG_cache = np.concatenate([[1], np.cumsum(self.discount_template)])

        # Correct predictions only
        denominator = y_pred.multiply(y_true)
        # Denominator: log2(rank_i + 1)
        denominator.data = np.log2(denominator.data + 1)
        # Binary relevance
        # Numerator: rel_i
        numerator = y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        per_user_dcg = dcg.sum(axis=1)

        hist_len = y_true.sum(axis=1).astype(np.int32)
        hist_len[hist_len > self.K] = self.K

        self._scores = sparse_divide_nonzero(
            csr_matrix(per_user_dcg),
            csr_matrix(self.IDCG_cache[hist_len]),
        )

        logger.debug(f"NDCGK compute complete - {self.name}")
