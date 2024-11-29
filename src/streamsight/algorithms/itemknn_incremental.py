import logging
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix import InteractionMatrix
from streamsight.utils.util import add_rows_to_csr_matrix


logger = logging.getLogger(__name__)


class ItemKNNIncremental(ItemKNN):
    """Incremental version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to allow for incremental updates
    to the model. The incremental updates are done by updating the historical
    data with the new data by appending the new data to the historical data.
    """

    def __init__(self, K=10, pad_with_popularity=True):
        super().__init__(K=K)
        self.pad_with_popularity = pad_with_popularity
        self.training_data: csr_matrix = None

    def append_training_data(self, X: csr_matrix):
        """Append a new interaction matrix to the historical data.

        :param X: The new interaction matrix
        :type X: csr_matrix
        """
        if self.training_data is None:
            return
        X_prev: csr_matrix = self.training_data.copy()
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
        self.training_data = X_prev + X

    def _fit(self, X: csr_matrix) -> "ItemKNNIncremental":
        """Fit a cosine similarity matrix from item to item."""
        if self.training_data is None:
            self.training_data = X.copy()
        else:
            self.append_training_data(X)
        super()._fit(self.training_data)
        return self

    def _predict(self, X: csr_matrix, predict_im: InteractionMatrix) -> csr_matrix:
        """Predict the K most similar items for each item using the latest data."""
        X_pred = super()._predict(self.training_data)
        # ID indexing starts at 0, so max_id + 1 is the number of unique IDs
        max_user_id = predict_im.max_user_id + 1
        max_item_id = predict_im.max_item_id + 1
        intended_shape = (
            max(max_user_id, X.shape[0]),
            max(max_item_id, X.shape[1]),
        )

        predict_frame = predict_im._df

        if X_pred.shape == intended_shape:
            return X_pred

        known_user_id, known_item_id = X_pred.shape
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0] - known_user_id)
        logger.debug(f"Padding user ID in range({known_user_id}, {intended_shape[0]}) with items")
        to_predict = predict_frame.value_counts("uid")

        if self.pad_with_popularity:
            popular_items = self.get_popularity_scores(X)
            for user_id in to_predict.index:
                if user_id >= known_user_id:
                    X_pred[user_id, :] = popular_items
        else:
            row = []
            col = []
            for user_id in to_predict.index:
                if user_id >= known_user_id:
                    row += [user_id] * to_predict[user_id]
                    col += self.rand_gen.integers(0, known_item_id, to_predict[user_id]).tolist()
            pad = csr_matrix((np.ones(len(row)), (row, col)), shape=intended_shape)
            X_pred += pad
        logger.debug(f"Padding by {self.name} completed")
        return X_pred

    def get_popularity_scores(self, X: csr_matrix):
        """Pad the predictions with popular items for users that are not in the training data."""
        interaction_counts = X.sum(axis=0).A[0]
        sorted_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if num_items < self.K:
            warn("K is larger than the number of items.", UserWarning)

        K = min(self.K, num_items)
        ind = np.argpartition(sorted_scores, -K)[-K:]
        a = np.zeros(X.shape[1])
        a[ind] = sorted_scores[ind]

        return a
