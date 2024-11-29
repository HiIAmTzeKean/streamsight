# Adopted from RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging

from scipy.sparse import csr_matrix, lil_matrix

from streamsight.metrics.base import ElementwiseMetricK

logger = logging.getLogger(__name__)


class HitK(ElementwiseMetricK):
  """Computes the number of hits in a list of Top-K recommendations.

  A hit is counted when a recommended item in the top K for this user was interacted with.

  Detailed :attr:`results` show which of the items in the list of Top-K recommended items
  were hits and which were not.

    
  :param K: Size of the recommendation list consisting of the Top-K item predictions.
  :type K: int

  This code is adapted from RecPack :cite:`recpack`
  """

  def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
    # log number of users and ground truth interactions
    logger.debug(f"HitK compute started - {self.name}")
    logger.debug(f"Number of users: {y_true.shape[0]}")
    logger.debug(f"Number of ground truth interactions: {y_true.nnz}")

    scores = lil_matrix(y_pred_top_K.shape)

    # Elementwise multiplication of top K predicts and true interactions
    scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1

    scores = scores.tocsr()

    self._scores = scores

    logger.debug(f"HitK compute complete - {self.name}")
