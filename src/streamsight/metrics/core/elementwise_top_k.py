import logging
from warnings import warn

import numpy as np

from .top_k import MetricTopK


logger = logging.getLogger(__name__)


class ElementwiseMetricK(MetricTopK):
    """Base class for all elementwise metrics that can be calculated for
    each user-item pair in the Top-K recommendations.

    :attr:`results` contains an entry for each user-item pair.

    Examples are: HitK

    This code is adapted from RecPack :cite:`recpack`

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    # TODO to fix this function
    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        if not self._is_computed:
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            warn(UserWarning("No scores were computed. Returning empty dict."))
            return dict(zip(self.col_names, (np.array([]), np.array([]))))

        scores = self._scores.toarray().reshape(-1)
        unique_users, inv = np.unique(self._user_id_sequence_array, return_inverse=True)

        # Sum hits per user
        sum_ones = np.zeros(len(unique_users))
        np.add.at(sum_ones, inv, scores)

        # Count recommendations per user
        count_all = np.zeros(len(unique_users))
        np.add.at(count_all, inv, 1)

        # aggregated score per user
        agg_score = sum_ones / count_all

        return dict(zip(self.col_names, (unique_users, agg_score)))


    @property
    def macro_result(self) -> None | float:
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

        scores = self._scores.toarray().reshape(-1)
        unique_users, inv = np.unique(self._user_id_sequence_array, return_inverse=True)
        # get all users that was recommended at least a relevant item
        sum_ones = np.zeros(len(unique_users))
        np.add.at(sum_ones, inv, scores)
        # Convert to binary: 1 if at least 1 hit, 0 otherwise
        binary_hits = (sum_ones > 0).astype(int)
        # Fraction of users with at least 1 hit
        return binary_hits.mean().item()
