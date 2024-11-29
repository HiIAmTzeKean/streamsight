import logging
from typing import Optional

import pandas as pd
from scipy.sparse import csr_matrix
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix import InteractionMatrix

logger = logging.getLogger(__name__)


class ItemKNNStatic(ItemKNN):
    """Static version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to only fit the model once. :meth:`fit` will only
    fit the model once and will not update the model with new data. The purpose
    is to make the training data static and not update the model with new data.
    """

    def __init__(self, K=10):
        super().__init__(K)
        self.fit_complete = False

    def fit(self, X: InteractionMatrix) -> "Algorithm":
        if self.fit_complete:
            return self

        super().fit(X)
        return self

    def _predict(
        self, X: csr_matrix, predict_frame: Optional[pd.DataFrame] = None
    ) -> csr_matrix:
        num_item, _ = self.similarity_matrix_.shape
        # reduce X to only the items that are in the similarity matrix
        X = X[:, :num_item]
        return super()._predict(X)
