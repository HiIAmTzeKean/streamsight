from typing import Optional
import pandas as pd

import numpy as np
from scipy.sparse import csr_matrix
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.utils import get_top_K_values

class Random(Algorithm):
    """Random recommendation for users.
    
    The Random algorithm recommends K random items to all users in the predict frame.
    
    :param K: Number of items to recommendation, defaults to 10
    :type K: int, optional
    :param seed: Seed for the random number, defaults to None
    :type seed: int, optional
    """

    def __init__(self, K: Optional[int] = 10, seed: Optional[int] = None):
        super().__init__()
        self.K = K
        # overwrite seed if provided
        if seed:
            self.seed = 42
        self.rand_gen = np.random.default_rng(seed=self.seed)

    def _fit(self, X: csr_matrix) -> "Random":
        self.fit_complete_ = True
        return self

    def _predict(self, X: csr_matrix, predict_frame:Optional[pd.DataFrame]=None) -> csr_matrix:
        """Predict the top K items for each user in the predict frame.
        
        If the predict frame is not provided, an AttributeError is raised.

        :param X: _description_
        :type X: csr_matrix
        :param predict_frame: _description_, defaults to None
        :type predict_frame: Optional[pd.DataFrame], optional
        :raises AttributeError: _description_
        :return: _description_
        :rtype: csr_matrix
        """
        if predict_frame is None:
            raise AttributeError("Predict frame with requested ID is required for Random algorithm")
        
        known_item_id = X.shape[1]
        
        # predict_frame contains (user_id, -1) pairs
        max_user_id  = predict_frame["uid"].max() + 1 
        intended_shape = (max(max_user_id, X.shape[0]), known_item_id)
        
        to_predict = predict_frame.value_counts("uid")
        row = []
        col = []
        for user_id in to_predict.index:
            row += [user_id] * to_predict[user_id]
            col += self.rand_gen.integers(0, known_item_id, to_predict[user_id]).tolist()
        scores = csr_matrix((np.ones(len(row)), (row, col)), shape=intended_shape)
        
        # Get top K of allowed items per user
        X_pred = get_top_K_values(scores, K=self.K)

        return X_pred