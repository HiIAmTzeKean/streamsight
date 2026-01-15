import logging

import numpy as np

from .top_k import MetricTopK


logger = logging.getLogger(__name__)


class ListwiseMetricK(MetricTopK):
    """Base class for all listwise metrics that can be calculated for every Top-K recommendation list,
    i.e. one value for each user.
    Examples are: PrecisionK, RecallK, DCGK, NDCGK.

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        """User level results for the metric.

        Contains an entry for every user.

        :return: The results DataFrame with columns: user_id, score
        :rtype: pd.DataFrame
        """
        if not self._is_computed:
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            logger.warning(UserWarning("No scores were computed. Returning empty dict."))
            return dict(zip(self.col_names, (np.array([]), np.array([]))))

        scores = self._scores.toarray().reshape(-1)

        unique_users, inv = np.unique(self._user_id_sequence_array, return_inverse=True)

        # sum of scores per user
        sum_ones = np.zeros(len(unique_users))
        np.add.at(sum_ones, inv, scores)

        # count per user
        count_all = np.zeros(len(unique_users))
        np.add.at(count_all, inv, 1)

        # aggregated score per user
        agg_score = sum_ones / count_all

        return dict(zip(self.col_names, (unique_users, agg_score)))

    @property
    def macro_result(self) -> None | float:
        """Global metric value obtained by taking the average over all users.

        :raises ValueError: If the metric has not been calculated yet.
        :return: The global metric value.
        :rtype: float, optional
        """
        if not self._is_computed:
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            logger.warning(UserWarning("No scores were computed. Returning Null value."))
            return None
        elif self._scores.size == 0:
            logger.warning(
                UserWarning(
                    f"All predictions were off or the ground truth matrix was empty during compute of {self.identifier}."
                )
            )
            return 0
        return self._scores.mean().item()
