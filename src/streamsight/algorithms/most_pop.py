from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from streamsight.algorithms import Algorithm
from streamsight.matrix.interaction_matrix import InteractionMatrix

class MostPop(Algorithm):
    """A popularity-based algorithm with based on MostPop by accumulating data from earlier time windows.

    :param K: Number of items to recommend, defaults to 200
    :type K: int, optional
    """

    def __init__(self, K: int = 200):
        super().__init__()
        self.K = K
        self.historical_data: list[csr_matrix] = []  # Store all historical training data
        self.num_items = 0  # Track the maximum number of items seen so far
    
    def _pad_matrix(self, matrix: csr_matrix, new_num_items: int) -> csr_matrix:
        """
        Pad a sparse matrix with zero columns to match the new number of items.

        :param matrix: The matrix to pad
        :type matrix: csr_matrix
        :param new_num_items: The target number of columns
        :type new_num_items: int
        :return: The padded matrix
        :rtype: csr_matrix
        """
        if matrix.shape[1] >= new_num_items:
            return matrix
        padding = csr_matrix((matrix.shape[0], new_num_items - matrix.shape[1]))
        return csr_matrix(np.hstack([matrix.toarray(), padding.toarray()]))

    def _expand_historical_data(self, new_num_items: int):
        """
        Expand all matrices in historical_data to match the new number of items.

        :param new_num_items: The updated number of items
        :type new_num_items: int
        """
        for i in range(len(self.historical_data)):
            if self.historical_data[i].shape[1] < new_num_items:
                self.historical_data[i] = self._pad_matrix(self.historical_data[i], new_num_items)

    def _fit(self, X: csr_matrix) -> "MostPop":
        """
        Fit the model by applying decay to historical data and adding new data.

        :param X: Interaction matrix (users x items) for the current window
        :type X: csr_matrix
        """
        # Update the maximum number of items
        new_num_items = X.shape[1]
        if new_num_items > self.num_items:
            self._expand_historical_data(new_num_items)
            self.num_items = new_num_items

        # Append the new matrix (ensure it has the correct number of items)
        if X.shape[1] < self.num_items:
            X = self._pad_matrix(X, self.num_items)

        # Append new data to historical data
        self.historical_data.append(X)

        # Initialize decayed scores
        num_items = X.shape[1]
        if num_items < self.K:
            warn("K is larger than the number of items.", UserWarning)

        interaction_counts = np.zeros(num_items)

        for matrix in self.historical_data:
            interaction_counts += matrix.sum(axis=0).A[0]

        normalized_scores = interaction_counts / interaction_counts.max()
        
        K = min(self.K, num_items)
        ind = np.argpartition(normalized_scores, -K)[-K:]
        a = np.zeros(num_items)
        a[ind] = normalized_scores[ind]
        self.sorted_scores_ = a
        return self

    def _predict(self, X: csr_matrix,  predict_im: InteractionMatrix) -> csr_matrix:
        """
        Predict the K most popular item for each user.

        """    
        if predict_im is None:
            raise AttributeError("Predict frame with requested ID is required for Popularity algorithm")

        predict_frame = predict_im._df

        users = predict_frame["uid"].unique().tolist()
        known_item_id = X.shape[1]
        
        # predict_frame contains (user_id, -1) pairs
        max_user_id  = predict_frame["uid"].max() + 1 
        intended_shape = (max(max_user_id, X.shape[0]), known_item_id)

        X_pred = lil_matrix(intended_shape)
        X_pred[users] = self.sorted_scores_

        return X_pred.tocsr()