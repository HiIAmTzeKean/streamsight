import logging

import numpy as np
from scipy.sparse import csr_matrix

from .interaction_matrix import InteractionMatrix


logger = logging.getLogger(__name__)


class PredictionMatrix(InteractionMatrix):

    @property
    def values(self) -> csr_matrix:
        """All user-item interactions as a sparse matrix of size (|`global_users`|, |`global_items`|).

        Each entry is the number of interactions between that user and item.
        If there are no interactions between a user and item, the entry is 0.

        :return: Interactions between users and items as a csr_matrix.
        :rtype: csr_matrix
        """
        raise ValueError("PredictionMatrix does not support values property. ")
