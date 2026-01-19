import logging
from typing import Self

from scipy.sparse import csr_matrix, hstack, vstack

from ..base import PopularityPaddingMixin, TopKItemSimilarityMatrixAlgorithm
from .itemknn import ItemKNN


logger = logging.getLogger(__name__)


class ItemKNNIncremental(ItemKNN):
    """Incremental version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to allow for incremental updates
    to the model. The incremental updates are done by updating the historical
    data with the new data by appending the new data to the historical data.
    """

    IS_BASE: bool = False

    def __init__(self, K: int = 10, pad_with_popularity: bool = True) -> None:
        PopularityPaddingMixin.__init__(self, pad_with_popularity=pad_with_popularity)
        TopKItemSimilarityMatrixAlgorithm.__init__(self, K=K)
        self.X_: None | csr_matrix = None

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
        """Fit a cosine similarity matrix from item to item."""
        if self.X_ is not None:
            self._append_training_data(X)
            super()._fit(self.X_)
        else:
            super()._fit(X)
        return self
