
import logging
from typing import Self

import numpy as np
from scipy.sparse import csr_matrix

from ...matrix import PredictionMatrix
from ..base import PopularityPaddingMixin, TopKAlgorithm


logger = logging.getLogger(__name__)


class RecentPopularity(TopKAlgorithm, PopularityPaddingMixin):
    """A popularity-based algorithm which only considers popularity of the latest train data."""

    IS_BASE: bool = False

    def _fit(self, X: csr_matrix) -> Self:
        self.sorted_scores_ = self.get_popularity_scores(X)
        return self

    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        """
        Predict the K most popular item for each user using only data from the latest window.
        """
        intended_shape = (X.get_prediction_data().num_interactions, X.user_item_shape[1])

        # Vectorized: repeat the sorted scores for each prediction row
        data = np.tile(self.sorted_scores_, (intended_shape[0], 1))
        X_pred = csr_matrix(data)

        return X_pred
