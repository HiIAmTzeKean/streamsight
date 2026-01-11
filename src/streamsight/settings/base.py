import logging
import time
from abc import abstractmethod
from typing import Any, Self, Union
from warnings import warn

from streamsight.matrix import InteractionMatrix
from ..models import BaseModel, ParamMixin
from .exception import EOWSettingError
from .processor import PredictionDataProcessor
from .schema import SplitResult


logger = logging.getLogger(__name__)


class Setting(BaseModel, ParamMixin):
    """Base class for defining an evaluation setting.

    Core Attributes:
    - background_data: Data used for inital training of model. Interval is [0, background_t).
    - unlabeled_data: List of unlabeled data. Each element is an InteractionMatrix
        object of interval [0, t).
    - ground_truth_data: List of ground truth data. Each element is an
        InteractionMatrix object of interval [t, t + window_size).
    - incremental_data: List of data used to incrementally update the model.
        Each element is an InteractionMatrix object of interval [t, t + window_size).
        Unique to SlidingWindowSetting.
    - data_timestamp_limit: List of timestamps that the splitter will slide over.

    We will use `background_data` as the initial training set, `incremental_data` as the data
    to incrementally update the model. However, for public methods, we will refer to both as
    `training_data` to avoid confusion.

    Args:
        seed: Seed for randomization. Defaults to 42.
    """

    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        """Initialize the setting.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.prediction_data_processor = PredictionDataProcessor()
        self._num_split_set = 1

        self._sliding_window_setting = False
        self._split_complete = False
        """Number of splits created from sliding window. Defaults to 1 (no splits on training set)."""
        self._num_full_interactions: int
        self._unlabeled_data: InteractionMatrix | list[InteractionMatrix]
        self._ground_truth_data: InteractionMatrix | list[InteractionMatrix]
        """Data containing the ground truth interactions to :attr:`_unlabeled_data`. If :class:`SlidingWindowSetting`, then it will be a list of :class:`InteractionMatrix`."""
        self._incremental_data: list[InteractionMatrix]
        """Data that is used to incrementally update the model. Unique to :class:`SlidingWindowSetting`."""
        self._background_data: InteractionMatrix
        """Data used as the initial set of interactions to train the model."""
        self._t_window: Union[None, int, list[int]]
        """This is the upper timestamp of the window in split. The actual interaction might have a smaller timestamp value than this because this will is the t cut off value."""
        self.n_seq_data: int
        """Number of last sequential interactions to provide in :attr:`unlabeled_data` as data for model to make prediction."""
        self.top_K: int
        """Number of interaction per user that should be selected for evaluation purposes in :attr:`ground_truth_data`."""

    def __str__(self) -> str:
        attrs = self.params
        return f"{self.__class__.__name__}({', '.join((f'{k}={v}' for k, v in attrs.items()))})"

    def get_params(self) -> dict[str, Any]:
        """Get the parameters of the setting."""
        # Get all instance attributes that don't start with underscore
        # and are not special attributes
        exclude_attrs = {"prediction_data_processor"}

        params = {}
        for attr_name, attr_value in vars(self).items():
            if not attr_name.startswith("_") and attr_name not in exclude_attrs:
                params[attr_name] = attr_value

        return params

    @property
    def identifier(self) -> str:
        """Name of the setting."""
        # return f"{super().identifier[:-1]},K={self.K})"
        paramstring = ",".join((f"{k}={v}" for k, v in self.params.items() if v is not None))
        return self.name + "(" + paramstring + ")"

    @abstractmethod
    def _split(self, data: InteractionMatrix) -> None:
        """Split data according to the setting.

        This abstract method must be implemented by concrete setting classes
        to split data into background_data, ground_truth_data, and unlabeled_data.

        Args:
            data: Interaction matrix to be split.
        """

    def split(self, data: InteractionMatrix) -> None:
        """Split data according to the setting.

        Calling this method changes the state of the setting object to be ready
        for evaluation. The method splits data into background_data, ground_truth_data,
        and unlabeled_data.

        Note:
            SlidingWindowSetting will have an additional attribute incremental_data.

        Args:
            data: Interaction matrix to be split.
        """
        logger.debug("Splitting data...")
        self._num_full_interactions = data.num_interactions
        start = time.time()
        self._split(data)
        end = time.time()
        logger.info(f"{self.name} data split - Took {end - start:.3}s")

        logger.debug("Checking split attribute and sizes.")
        self._check_split()

        self._split_complete = True
        logger.info(f"{self.name} data split complete.")

    def _check_split_complete(self) -> None:
        """Check if the setting is ready for evaluation.

        Raises:
            KeyError: If the setting has not been split yet.
        """
        if not self.is_ready:
            raise KeyError("Setting has not been split yet. Call split() method before accessing the property.")

    @property
    def num_split(self) -> int:
        """Get number of splits created from dataset.

        This property defaults to 1 (no splits on training set) for typical settings.
        For SlidingWindowSetting, this is typically greater than 1 if there are
        multiple splits created from the sliding window.

        Returns:
            Number of splits created from dataset.
        """
        return self._num_split_set

    @property
    def is_ready(self) -> bool:
        """Check if setting is ready for evaluation.

        Returns:
            True if the setting has been split and is ready to use.
        """
        return self._split_complete

    @property
    def is_sliding_window_setting(self) -> bool:
        """Check if setting is SlidingWindowSetting.

        Returns:
            True if this is a SlidingWindowSetting instance.
        """
        return self._sliding_window_setting

    @property
    def background_data(self) -> InteractionMatrix:
        """Get background data for initial model training.

        Returns:
            InteractionMatrix of training interactions.
        """
        self._check_split_complete()
        return self._background_data

    @property
    def t_window(self) -> Union[None, int, list[int]]:
        """Get the upper timestamp of the window in split.

        In settings that respect the global timeline, returns a timestamp value.
        In `SlidingWindowSetting`, returns a list of timestamp values.
        In settings like `LeaveNOutSetting`, returns None.

        Returns:
            Timestamp limit for the data (int, list of ints, or None).
        """
        self._check_split_complete()
        return self._t_window

    @property
    def unlabeled_data(self) -> InteractionMatrix | list[InteractionMatrix]:
        """Get unlabeled data for model predictions.

        Contains the user/item ID for prediction along with previous sequential
        interactions. Used to make predictions on ground truth data.

        Returns:
            Single InteractionMatrix or list of InteractionMatrix for sliding window setting.
        """
        self._check_split_complete()

        if not self._sliding_window_setting:
            return self._unlabeled_data
        return self._unlabeled_data

    @property
    def ground_truth_data(self) -> InteractionMatrix | list[InteractionMatrix]:
        """Get ground truth data for model evaluation.

        Contains the actual interactions of user-item that the model should predict.

        Returns:
            Single InteractionMatrix or list of InteractionMatrix for sliding window.
        """
        self._check_split_complete()

        if not self._sliding_window_setting:
            return self._ground_truth_data
        return self._ground_truth_data

    @property
    def incremental_data(self) -> list[InteractionMatrix]:
        """Get data for incrementally updating the model.

        Only available for SlidingWindowSetting.

        Returns:
            List of InteractionMatrix objects for incremental updates.

        Raises:
            AttributeError: If setting is not SlidingWindowSetting.
        """
        self._check_split_complete()

        if not self._sliding_window_setting:
            raise AttributeError("Incremental data is only available for sliding window setting.")
        return self._incremental_data

    def _check_split(self) -> None:
        """Checks that the splits have been done properly.

        Makes sure all expected attributes are set.
        """
        logger.debug("Checking split attributes.")
        assert hasattr(self, "_background_data") and self._background_data is not None

        assert (hasattr(self, "_unlabeled_data") and self._unlabeled_data is not None) or (
            hasattr(self, "_unlabeled_data") and self._unlabeled_data is not None
        )

        assert (hasattr(self, "_ground_truth_data") and self._ground_truth_data is not None) or (
            hasattr(self, "_ground_truth_data") and self._ground_truth_data is not None
        )
        logger.debug("Split attributes are set.")

        self._check_size()

    def _check_size(self) -> None:
        """
        Warns user if any of the sets is unusually small or empty
        """
        logger.debug("Checking size of split sets.")

        def check_ratio(name, count, total, threshold) -> None:
            if check_empty(name, count):
                return

            if (count + 1e-9) / (total + 1e-9) < threshold:
                warn(UserWarning(f"{name} resulting from {self.name} is unusually small."))

        def check_empty(name, count) -> bool:
            if count == 0:
                warn(UserWarning(f"{name} resulting from {self.name} is empty (no interactions)."))
                return True
            return False

        n_background = self._background_data.num_interactions
        # check_empty("Background data", n_background)
        check_ratio("Background data", n_background, self._num_full_interactions, 0.05)

        if not self._sliding_window_setting:
            n_unlabel = self._unlabeled_data.num_interactions
            n_ground_truth = self._ground_truth_data.num_interactions

            check_empty("Unlabeled data", n_unlabel)
            # check_empty("Ground truth data", n_ground_truth)
            check_ratio("Ground truth data", n_ground_truth, n_unlabel, 0.05)

        else:
            for dataset_idx in range(self._num_split_set):
                n_unlabel = self._unlabeled_data[dataset_idx].num_interactions
                n_ground_truth = self._ground_truth_data[dataset_idx].num_interactions

                check_empty(f"Unlabeled data[{dataset_idx}]", n_unlabel)
                check_empty(f"Ground truth data[{dataset_idx}]", n_ground_truth)
        logger.debug("Size of split sets are checked.")

    def restore(self, n: int = 0) -> None:
        """Restore last run.

        Args:
            n: Iteration number to restore to. If None, restores to beginning.
        """
        logger.debug(f"Restoring setting to iteration {n}")
        self.current_index = n

    def __iter__(self) -> Self:
        """Iterate over splits in the setting.

        Resets the index and returns self as the iterator.
        Yields a SplitResult for each split: {'unlabeled', 'ground_truth', 't_window', 'incremental'}.
        """
        self.current_index = 0
        return self

    def __next__(self) -> SplitResult:
        """Get the next split.

        Returns:
            SplitResult with split data.

        Raises:
            EOWSettingError: If no more splits.
        """
        if self.current_index >= self.num_split:
            raise EOWSettingError("No more splits available, EOW reached.")

        if self._sliding_window_setting:
            if not (
                isinstance(self._unlabeled_data, list)
                and isinstance(self._ground_truth_data, list)
                and isinstance(self._t_window, list)
            ):
                raise ValueError("Expected list of InteractionMatrix for sliding window setting.")
            result = SplitResult(
                unlabeled=self._unlabeled_data[self.current_index],
                ground_truth=self._ground_truth_data[self.current_index],
                t_window=self._t_window[self.current_index],
                incremental=(
                    self._incremental_data[self.current_index - 1]
                    if self.current_index < len(self._incremental_data) and self.current_index > 1
                    else None
                ),
            )
        else:
            if (
                isinstance(self._unlabeled_data, list)
                or isinstance(self._ground_truth_data, list)
                or isinstance(self._t_window, list)
            ):
                raise ValueError("Expected single InteractionMatrix for non-sliding window setting.")
            result = SplitResult(
                unlabeled=self._unlabeled_data,
                ground_truth=self._ground_truth_data,
                t_window=self._t_window,
                incremental=None,
            )

        self.current_index += 1
        return result

    def get_split_at(self, index: int) -> SplitResult:
        """Get the split data at a specific index.

        Args:
            index: The index of the split to retrieve.

        Returns:
            SplitResult with keys: 'unlabeled', 'ground_truth', 't_window', 'incremental'.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index > self.num_split:
            raise IndexError(f"Index {index} out of range for {self.num_split} splits")

        if self._sliding_window_setting:
            if not (
                isinstance(self._unlabeled_data, list)
                and isinstance(self._ground_truth_data, list)
                and isinstance(self._t_window, list)
            ):
                raise ValueError("Expected list of InteractionMatrix for sliding window setting.")
            result = SplitResult(
                unlabeled=self._unlabeled_data[index],
                ground_truth=self._ground_truth_data[index],
                # TODO change this variable to training_data when refactoring
                incremental=(
                    self._incremental_data[index - 1] if index < len(self._incremental_data) and index > 0 else None
                ),
                t_window=self._t_window[index],
            )
        else:
            if index != 0:
                raise IndexError("Non-sliding setting has only one split at index 0")
            if (
                isinstance(self._unlabeled_data, list)
                or isinstance(self._ground_truth_data, list)
                or isinstance(self._t_window, list)
            ):
                raise ValueError("Expected single data for non-sliding setting.")
            result = SplitResult(
                unlabeled=self._unlabeled_data,
                ground_truth=self._ground_truth_data,
                incremental=None,
                t_window=self._t_window,
            )

        return result
