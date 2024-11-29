import logging

from streamsight.matrix import InteractionMatrix
from .timestamp import TimestampSplitter


logger = logging.getLogger(__name__)


class NLastInteractionTimestampSplitter(TimestampSplitter):
    """Splits with n last interactions based on a timestamp.

    Splits the data into unlabeled and ground truth data based on a timestamp.
    Historical data contains last `n_seq_data` interactions before the timestamp `t`
    and the future interaction contains interactions after the timestamp `t`.


    Attributes:
        past_interaction: List of unlabeled data. Interval is `[0, t)`.
        - future_interaction: Data used for training the model.
            Interval is `[t, t+t_upper)` or `[t,inf]`.
        n_seq_data: Number of last interactions to provide as data for model to make prediction.
            These interactions are past interactions from before the timestamp `t`.

    Args:
        t: Timestamp to split on in seconds since epoch.
        t_upper: Seconds past t. Upper bound on the timestamp
            of interactions. Defaults to None (infinity).
        n_seq_data: Number of last interactions to provide as data
            for model to make prediction. Defaults to 1.
        include_all_past_data: If True, include all past data in the past_interaction.
            Defaults to False.
    """

    def __init__(
        self,
        t: int,
        t_upper: None | int = None,
        n_seq_data: int = 1,
        include_all_past_data: bool = False,
    ) -> None:
        super().__init__(t=t, t_lower=None, t_upper=t_upper)
        self.n_seq_data = n_seq_data
        self.include_all_past_data = include_all_past_data

    def update_split_point(self, t: int) -> None:
        logger.debug(f"{self.identifier} - Updating split point to t={t}")
        self.t = t

    def split(self, data: InteractionMatrix) -> tuple[InteractionMatrix, InteractionMatrix]:
        """Splits data such that the following definition holds:

        - past_interaction: List of unlabeled data. Interval is `[0, t)`.
        - future_interaction: Data used for training the model.
            Interval is `[t, t+t_upper)` or `[t,inf]`.

        Args:
            data: Interaction matrix to be split. Must contain timestamps.

        Returns:
            A 2-tuple containing the `past_interaction` and `future_interaction` matrices.
        """
        if self.t_upper is None:
            future_interaction = data.timestamps_gte(self.t)
        else:
            future_interaction = data.timestamps_lt(self.t + self.t_upper).timestamps_gte(self.t)

        if self.include_all_past_data:
            past_interaction = data.timestamps_lt(self.t)
        else:
            past_interaction = data.get_users_n_last_interaction(
                self.n_seq_data, self.t, future_interaction.user_ids
            )

        logger.debug(f"{self.identifier} has complete split")
        return past_interaction, future_interaction
