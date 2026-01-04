import logging
from typing import Self

from streamsight.matrix import InteractionMatrix
from .base import Algorithm
from .itemknn import ItemKNN


logger = logging.getLogger(__name__)


class ItemKNNStatic(ItemKNN):
    """Static version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to only fit the model once. :meth:`fit` will only
    fit the model once and will not update the model with new data. The purpose
    is to make the training data static and not update the model with new data.
    """

    IS_BASE: bool = False

    def __init__(self, K: int = 10) -> None:
        self._is_fitted = False
        super().__init__(K)

    def fit(self, X: InteractionMatrix) -> Self:
        if self._is_fitted:
            return self

        super().fit(X)
        self._is_fitted = True
        return self
