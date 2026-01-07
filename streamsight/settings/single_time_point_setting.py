import logging
from typing import Literal, Optional
from warnings import warn

import numpy as np

from streamsight.matrix import InteractionMatrix, TimestampAttributeMissingError
from streamsight.settings import Setting
from streamsight.settings.splitters import (
    NPastInteractionTimestampSplitter,
    TimestampSplitter,
)

logger = logging.getLogger(__name__)


class SingleTimePointSetting(Setting):
    """Single time point setting for data split.

    :param background_t: Time point to split the data into background and evaluation data. Split will be from ``[0, t)``
    :type background_t: int
    :param n_seq_data: Number of last sequential interactions to provide as
        unlabeled data for model to make prediction.
    :type n_seq_data: int, optional
    :param top_K: Number of interaction per user that should be selected for evaluation purposes.
    :type top_K: int, optional
    :param t_upper: Upper bound on the timestamp of interactions.
        Defaults to maximal integer value (acting as infinity).
    :type t_upper: int, optional
    :param seed: Seed for randomization parts of the scenario.
        Timed scenario is deterministic, so changing seed should not matter.
        Defaults to None, so random seed will be generated.
    :type seed: int, optional
    """

    def __init__(
        self,
        background_t: int,
        n_seq_data: int = 1,
        top_K: int = 1,
        t_upper: int = np.iinfo(np.int32).max,
        include_all_past_data: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.t = background_t
        """Seconds before `t` timestamp value to be used in `background_set`."""
        self.t_upper = t_upper
        """Seconds after `t` timestamp value to be used in `ground_truth_data`."""
        self.n_seq_data = n_seq_data
        self.top_K = top_K

        logger.info(
            f"Splitting data at time {background_t} with t_upper interval {t_upper}"
        )

        self._background_splitter = TimestampSplitter(
            background_t, None, t_upper
        )
        self._splitter = NPastInteractionTimestampSplitter(
            background_t, t_upper, n_seq_data, include_all_past_data
        )
        self._t_window = background_t

    def _split(self, data: InteractionMatrix):
        """Splits your dataset into a training, validation and test dataset
            based on the timestamp of the interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        if not data.has_timestamps:
            raise TimestampAttributeMissingError()
        if data.min_timestamp > self.t:
            warn(
                f"Splitting at time {self.t} is before the first timestamp"
                " in the data. No data will be in the training set."
            )

        self._background_data, _ = self._background_splitter.split(data)
        past_interaction, future_interaction = self._splitter.split(data)
        self._unlabeled_data, self._ground_truth_data = (
            self.prediction_data_processor.process(
                past_interaction, future_interaction, self.top_K
            )
        )
        
    @property
    def params(self):
        """Parameters of the setting."""
        return {
            "background_t": self.t,
            "t_upper": self.t_upper,
            "n_seq_data": self.n_seq_data,
            "top_K": self.top_K,
        }