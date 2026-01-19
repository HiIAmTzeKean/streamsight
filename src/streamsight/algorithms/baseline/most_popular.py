import logging
from typing import Self

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from ...matrix import PredictionMatrix
from ..base import PopularityPaddingMixin, TopKAlgorithm


logger = logging.getLogger(__name__)


class MostPopular(TopKAlgorithm, PopularityPaddingMixin):
    """A popularity-based algorithm that considers all historical data."""

    IS_BASE: bool = False
    X_: csr_matrix | None = None  # Store all historical training data

    def _append_training_data(self, X: csr_matrix) -> None:
        """Append a new interaction matrix to the historical data.

        Args:
            X (csr_matrix): Interaction matrix to append
        """
        if self.X_ is None:
            raise ValueError("No existing training data to append to.")
        X_prev: csr_matrix = self.X_.copy()
        new_num_rows = max(X_prev.shape[0], X.shape[0])
        new_num_cols = max(X_prev.shape[1], X.shape[1])
        # Pad the previous matrix
        if X_prev.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X_prev.shape[0], X_prev.shape[1]))
            X_prev = vstack([X_prev, row_padding])
        if X_prev.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X_prev.shape[0], new_num_cols - X_prev.shape[1]))
            X_prev = hstack([X_prev, col_padding])

        # Pad the current matrix
        if X.shape[0] < new_num_rows:  # Pad rows
            row_padding = csr_matrix((new_num_rows - X.shape[0], X.shape[1]))
            X = vstack([X, row_padding])
        if X.shape[1] < new_num_cols:  # Pad columns
            col_padding = csr_matrix((X.shape[0], new_num_cols - X.shape[1]))
            X = hstack([X, col_padding])

        # Merge data
        self.X_ = X_prev + X

    def _fit(self, X: csr_matrix) -> Self:
        if self.X_ is not None:
            self._append_training_data(X)
        else:
            self.X_ = X.copy()

        if not isinstance(self.X_, csr_matrix):
            raise ValueError("Training data is not initialized properly.")

        if self.X_.shape[1] < self.K:
            logger.warning("K is larger than the number of items.", UserWarning)

        self.sorted_scores_ = self.get_popularity_scores(self.X_)
        return self

    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        intended_shape = (X.get_prediction_data().num_interactions, X.user_item_shape[1])

        # Vectorized: repeat the sorted scores for each prediction row
        data = np.tile(self.sorted_scores_, (intended_shape[0], 1))
        X_pred = csr_matrix(data)

        return X_pred
