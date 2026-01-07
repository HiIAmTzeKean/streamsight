from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

from streamsight.algorithms.base import Algorithm

class Popularity(Algorithm):
    """Recommends K most popular items

    The Popularity algorithm recommends the K most popular items to all users
    in the predict frame. The popularity of an item is determined by the number
    of interactions it has received over the total number of interactions.

    :param K: Number of items to recommend, defaults to 10
    :type K: int, optional
    """

    def __init__(self, K: int = 10):
        super().__init__()
        self.K = K

    def _fit(self, X: csr_matrix) -> None:
        """Fit the Popularity algorithm.
        
        The popularity of an item is determined by the number of interactions it
        has received over the total number of interactions. The top K items are
        selected based on this popularity score.
        
        This code is adapted from RecPack :cite:`recpack`
        
        :param X: Interaction matrix
        :type X: csr_matrix
        """
        interaction_counts = X.sum(axis=0).A[0]
        sorted_scores = interaction_counts / interaction_counts.max()

        num_items = X.shape[1]
        if num_items < self.K:
            warn("K is larger than the number of items.", UserWarning)

        K = min(self.K, num_items)
        
        ind = np.argpartition(sorted_scores, -K)[-K:]
        a = np.zeros(X.shape[1])
        # set columns of top K to the scores
        a[ind] = sorted_scores[ind]
        self.sorted_scores_ = a

    def _predict(self, X: csr_matrix, predict_frame:Optional[pd.DataFrame]=None) -> csr_matrix:
        """Predict the K most popular items
        
        If the predict frame is not provided, an AttributeError is raised. The
        algorithm will recommend the K most popular items to all users in the
        predict frame.

        :param X: _description_
        :type X: csr_matrix
        :param predict_frame: _description_, defaults to None
        :type predict_frame: Optional[pd.DataFrame], optional
        :raises AttributeError: _description_
        :return: _description_
        :rtype: csr_matrix
        """
        if predict_frame is None:
            raise AttributeError("Predict frame with requested ID is required for Popularity algorithm")

        users = predict_frame["uid"].unique().tolist()
        
        known_item_id = X.shape[1]
        
        # predict_frame contains (user_id, -1) pairs
        max_user_id  = predict_frame["uid"].max() + 1 
        intended_shape = (max(max_user_id, X.shape[0]), known_item_id)

        X_pred = lil_matrix(intended_shape)
        X_pred[users] = self.sorted_scores_

        return X_pred.tocsr()
