import logging
from typing import Optional
from warnings import warn

import numpy as np
from tqdm import tqdm
from streamsight.matrix import (InteractionMatrix,
                                TimestampAttributeMissingError)
from streamsight.settings import Setting
from streamsight.settings.splitters import (NPastInteractionTimestampSplitter,
                                           TimestampSplitter)

logger = logging.getLogger(__name__)


class SlidingWindowSetting(Setting):
    """Sliding window setting for splitting data.

    The data is split into a background set and evaluation set. The evaluation set is defined by a sliding window
    that moves over the data. The window size is defined by the :data:`window_size` parameter. The evaluation set comprises of the
    unlabeled data and ground truth data stored in a list. The unlabeled data is the last :data:`n_seq_data` interactions of the users/item before the
    split point. The ground truth data is the interactions after the split point and spans :data:`window_size` seconds.

    Core attribute
    ====================
    - :attr:`background_data`: Data used for training the model. Interval is `[0, background_t)`.
    - :attr:`unlabeled_data`: List of unlabeled data. Each element is a :class:`InteractionMatrix` object of interval `[0, t)`.
    - :attr:`ground_truth_data`: List of ground truth data. Each element is a :class:`InteractionMatrix` object of interval `[t, t + window_size)`.
    - :attr:`data_timestamp_limit`: List of timestamps that the splitter will slide over the data.
    - :attr:`incremental_data`: List of data that is used to incrementally update the model. Each element is a :class:`InteractionMatrix` object of interval `[t, t + window_size)`.

    :param background_t: Time point to split the data into background and evaluation data. Split will be from `[0, t)`
    :type background_t: int
    :param window_size: Size of the window in seconds to slide over the data.
        Affects the incremental data being released to the model. If
        :param:`t_ground_truth_window` is not provided, ground truth data will also
        take this window.
    :type window_size: int, optional
    :param n_seq_data: Number of last sequential interactions to provide as
        unlabeled data for model to make prediction.
    :type n_seq_data: int, optional
    :param top_K: Number of interaction per user that should be selected for evaluation purposes.
    :type top_K: int, optional
    :param t_upper: Upper bound on the timestamp of interactions.
        Defaults to maximal integer value (acting as infinity).
    :type t_upper: int, optional
    :param t_ground_truth_window: Size of the window in seconds to slide over the data for ground truth data.
        If not provided, defaults to window_size during computation.
    :type t_ground_truth_window: int, optional
    :param seed: Seed for random number generator.
    :type seed: int, optional
    """

    def __init__(
        self,
        background_t: int,
        window_size: int = np.iinfo(np.int32).max,  # in seconds
        n_seq_data: int = 10,
        top_K: int = 10,
        t_upper: int = np.iinfo(np.int32).max,
        t_ground_truth_window: Optional[int] = None,
        seed: Optional[int] = None
    ):
        super().__init__(seed=seed)
        self._sliding_window_setting = True
        self.t = background_t
        self.window_size = window_size
        """Window size in seconds for splitter to slide over the data."""
        self.n_seq_data = n_seq_data
        self.top_K = top_K
        self.t_upper = t_upper
        """Upper bound on the timestamp of interactions. Defaults to maximal integer value (acting as infinity)."""
        
        if t_upper and t_upper < background_t:
            raise ValueError("t_upper must be greater than background_t")
        
        if t_ground_truth_window is None:
            t_ground_truth_window = window_size
        
        self.t_ground_truth_window = t_ground_truth_window
        
        self._background_splitter = TimestampSplitter(background_t, None, None)
        self._window_splitter = NPastInteractionTimestampSplitter(
            background_t, t_ground_truth_window, n_seq_data
        )

    def _split(self, data: InteractionMatrix):
        """Splits dataset into a background, unlabeled and ground truth data.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """
        if not data.has_timestamps:
            raise TimestampAttributeMissingError()
        if data.min_timestamp > self.t:
            warn(f"Splitting at time {self.t} is before the first "
                 "timestamp in the data. No data will be in the background(training) set."
            )
        if self.t_upper:
            data = data.timestamps_lt(self.t_upper)

        self._background_data, _ = self._background_splitter.split(data)
        self._ground_truth_data, self._unlabeled_data, self._t_window, self._incremental_data = [], [], [], []

        # sub_time is the subjugate time point that the splitter will slide over the data
        sub_time = self.t
        max_timestamp = data.max_timestamp

        pbar = tqdm(total=int((max_timestamp - sub_time) / self.window_size))
        while sub_time <= max_timestamp:
            self._t_window.append(sub_time)
            # the set used for eval will always have a timestamp greater than
            # data released such that it is unknown to the model
            self._window_splitter.update_split_point(sub_time)
            past_interaction, future_interaction = self._window_splitter.split(
                data
            )
            unlabeled_set, ground_truth = self.prediction_data_processor.process(past_interaction,
                                                                                 future_interaction,
                                                                                 self.top_K)
            self._unlabeled_data.append(unlabeled_set)
            self._ground_truth_data.append(ground_truth)
            
            self._incremental_data.append(future_interaction)
            
            sub_time += self.window_size
            pbar.update(1)
        pbar.close()

        self._num_split_set = len(self._unlabeled_data)
        logger.info(
            f"Finished split with window size {self.window_size} seconds. "
            f"Number of splits: {self._num_split_set} in total."
        )

    @property
    def params(self):
        """Parameters of the setting."""
        return {
            "background_t": self.t,
            "window_size": self.window_size,
            "n_seq_data": self.n_seq_data,
            "top_K": self.top_K,
            "t_upper": self.t_upper,
            "t_ground_truth_window": self.t_ground_truth_window
        }