import logging
from warnings import warn

import numpy as np

from streamsight.matrix import InteractionMatrix, TimestampAttributeMissingError
from streamsight.settings import Setting
from streamsight.settings.splitters import (
    NLastInteractionTimestampSplitter,
    TimestampSplitter,
)


logger = logging.getLogger(__name__)


class SingleTimePointSetting(Setting):
    """Single time point setting for data split.

    Splits an interaction dataset at a single timestamp into background
    (training) data and evaluation data. The evaluation data can be
    further processed to produce unlabeled inputs and ground-truth
    targets for model evaluation.

    Args:
        background_t: Time point to split the data. The background
            split covers interactions with timestamps in `[0, background_t)`.
        n_seq_data: Number of last sequential interactions
            to provide as input for prediction. Defaults to `1`.
        top_K: Number of interactions per user to select for
            evaluation purposes. Defaults to `1`.
        t_upper: Upper bound on the timestamp of
            interactions included in evaluation. Defaults to the maximum
            32-bit integer value (acts like infinity).
        include_all_past_data: If True, include all past
            interactions when constructing input sequences. Defaults to False.
        seed: Random seed for reproducible behavior.
            If None, a seed will be generated.
    """

    def __init__(
        self,
        background_t: int,
        n_seq_data: int = 1,
        top_K: int = 1,
        t_upper: int = np.iinfo(np.int32).max,
        include_all_past_data: bool = False,
        seed: None | int = None,
    ):
        super().__init__(seed=seed)
        self.t = background_t
        """Seconds before `t` timestamp value to be used in `background_set`."""
        self.t_upper = t_upper
        """Seconds after `t` timestamp value to be used in `ground_truth_data`."""
        self.n_seq_data = n_seq_data
        self.top_K = top_K

        logger.info(f"Splitting data at time {background_t} with t_upper interval {t_upper}")

        self._background_splitter = TimestampSplitter(
            background_t,
            None,
            t_upper,
        )
        self._splitter = NLastInteractionTimestampSplitter(
            background_t,
            t_upper,
            n_seq_data,
            include_all_past_data,
        )
        self._t_window = background_t

    def _split(self, data: InteractionMatrix) -> None:
        """Split the dataset by timestamp into background and evaluation sets.

        The method raises :class:`TimestampAttributeMissingError` when the
        provided :class:`InteractionMatrix` does not contain timestamp
        information. It will warn if the chosen split time is before the
        earliest timestamp in the data.

        Args:
            data: Interaction matrix to split. Must have timestamps.

        Raises:
            TimestampAttributeMissingError: If `data` has no timestamp attribute.
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
        self._unlabeled_data, self._ground_truth_data = self.prediction_data_processor.process(
            past_interaction,
            future_interaction,
            self.top_K,
        )

    @property
    def params(self) -> dict[str, int]:
        """Return a dictionary of the setting's parameters.

        Returns:
            Mapping of parameter names to their values.
        """
        return {
            "background_t": self.t,
            "t_upper": self.t_upper,
            "n_seq_data": self.n_seq_data,
            "top_K": self.top_K,
        }
