from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from streamsight.algorithms import Algorithm
from streamsight.matrix.interaction_matrix import InteractionMatrix

class RecentPopularity(Algorithm):
    """A popularity-based algorithm which only considers popularity of the latest train data.

    :param K: Number of items to recommend, defaults to 200
    :type K: int, optional
    """

    def __init__(self, K: int = 200):
        super().__init__()
        self.K = K


    def _fit(self, X: csr_matrix) -> "RecentPopularity":
        """
        Fit the model by applying decay to historical data and adding new data.

        :param X: Interaction matrix (users x items) for the current window
        :type X: csr_matrix
        """
        # Get popularity score for every item
        interaction_counts = X.sum(axis=0).A[0]
        sorted_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if num_items < self.K:
            warn("K is larger than the number of items.", UserWarning)

        K = min(self.K, num_items)
        ind = np.argpartition(sorted_scores, -K)[-K:]
        a = np.zeros(X.shape[1])
        a[ind] = sorted_scores[ind]
        self.sorted_scores_ = a
        return self

    def _predict(self, X: csr_matrix, predict_im: InteractionMatrix) -> csr_matrix:
        """
        Predict the K most popular item for each user using only train data from the latest window.
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

    