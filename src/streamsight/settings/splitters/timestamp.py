import logging

from streamsight.matrix import InteractionMatrix
from .base import Splitter


logger = logging.getLogger(__name__)


class TimestampSplitter(Splitter):
    """Split an interaction dataset by timestamp.

    The splitter divides the data into two parts:

    1. Interactions with timestamps in the interval `[t - t_lower, t)`,
       representing past interactions.
    2. Interactions with timestamps in the interval `[t, t + t_upper]`,
       representing future interactions.

    If `t_lower` or `t_upper` are not provided, they default to infinity,
    meaning the corresponding interval is unbounded on that side.

    Note that a user can appear in both the past and future interaction sets.

    Attributes:
        past_interaction (InteractionMatrix): Interactions in the interval
            `[0, t)`, representing unlabeled data for prediction.
        future_interaction (InteractionMatrix): Interactions in the interval
            `[t, t + t_upper)` or `[t, inf)`, used for training the model.

    Args:
        t: Timestamp to split on, in seconds since the Unix epoch.
        t_lower: Seconds before `t` to include in
            the past interactions. If None, the interval is unbounded.
            Defaults to None.
        t_upper: Seconds after `t` to include in
            the future interactions. If None, the interval is unbounded.
            Defaults to None.
    """

    def __init__(
        self,
        t: int,
        t_lower: None | int = None,
        t_upper: None | int = None,
    ) -> None:
        super().__init__()
        self.t = t
        self.t_lower = t_lower
        self.t_upper = t_upper

    def split(self, data: InteractionMatrix) -> tuple[InteractionMatrix, InteractionMatrix]:
        """Split the interaction data by timestamp.

        The method populates the `past_interaction` and `future_interaction`
        attributes with the corresponding subsets of the input data.

        Args:
            data: The interaction dataset to split.
                Must include timestamp information.

        Returns:
            A pair containing the past interactions and future interactions.
        """

        if self.t_lower is None:
            # timestamp < t
            past_interaction = data.timestamps_lt(self.t)
        else:
            # t-t_lower =< timestamp < t
            past_interaction = data.timestamps_lt(self.t).timestamps_gte(self.t - self.t_lower)

        if self.t_upper is None:
            # timestamp >= t
            future_interaction = data.timestamps_gte(self.t)
        else:
            # t =< timestamp < t + t_upper
            future_interaction = data.timestamps_gte(self.t).timestamps_lt(self.t + self.t_upper)

        logger.debug(f"{self.identifier} has complete split")

        return past_interaction, future_interaction
