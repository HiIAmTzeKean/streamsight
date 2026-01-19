from typing import Self

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ...matrix import PredictionMatrix
from ..base import TopKAlgorithm
from ..utils import get_top_K_values


class Random(TopKAlgorithm):
    """Random recommendation for users.

    The Random algorithm recommends K random items to all users in the predict frame.
    """
    IS_BASE: bool = False

    def _fit(self, X: csr_matrix) -> Self:  # noqa: ARG002
        self.fit_complete_ = True
        return self

    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        predict_ui_df = X.get_prediction_data()._df  # noqa: SLF001

        known_item_id = X.max_known_item_id
        intended_shape = (X.max_global_user_id, known_item_id)

        to_predict = pd.Series(predict_ui_df.uid.unique())
        to_predict = to_predict.sort_values(ignore_index=True)
        row = to_predict.values.repeat(self.K)
        total_items_to_predict = len(row)
        col = self.rand_gen.integers(0, known_item_id, total_items_to_predict)
        scores = csr_matrix((np.ones(total_items_to_predict), (row, col)), shape=intended_shape)

        # Get top K of allowed items per user
        X_pred = get_top_K_values(scores, K=self.K)
        X_pred = X_pred[predict_ui_df["uid"].values]
        return X_pred
