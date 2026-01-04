
import logging
from typing import Self

from scipy.sparse import csr_matrix, lil_matrix

from ..matrix import PredictionMatrix
from .base import PopularityPaddingMixin, TopKAlgorithm


logger = logging.getLogger(__name__)


class RecentPopularity(TopKAlgorithm, PopularityPaddingMixin):
    """A popularity-based algorithm which only considers popularity of the latest train data."""

    IS_BASE: bool = False

    def _fit(self, X: csr_matrix) -> Self:
        self.sorted_scores_ = self.get_popularity_scores(X)
        return self

    def _predict(self, X: PredictionMatrix) -> csr_matrix:
        """
        Predict the K most popular item for each user using only data from the latest window.
        """
        predict_ui_df = X.get_prediction_data()._df  # noqa: SLF001
        users = predict_ui_df["uid"].unique().tolist()

        # predict_ui_df contains (user_id, -1) pairs
        max_user_id = predict_ui_df["uid"].max() + 1
        intended_shape = (max(max_user_id, X.user_item_shape[0]), X.user_item_shape[1])

        X_pred = lil_matrix(intended_shape)
        X_pred[users] = self.sorted_scores_

        return csr_matrix(X_pred.tocsr())
