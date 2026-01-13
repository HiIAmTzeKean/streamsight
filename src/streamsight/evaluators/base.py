import logging
from typing import Literal

import pandas as pd
from scipy.sparse import csr_matrix

from ..matrix import PredictionMatrix
from ..registries import MetricEntry
from ..settings import EOWSettingError, Setting
from .accumulator import MetricAccumulator
from .util import MetricLevelEnum, UserItemBaseStatus


logger = logging.getLogger(__name__)


class EvaluatorBase(object):
    """Base class for evaluator.

    Provides the common methods and attributes for the evaluator classes. Should
    there be a need to create a new evaluator, it should inherit from this class.

    Args:
        metric_entries: List of metric entries to compute.
        setting: Setting object.
        ignore_unknown_user: Ignore unknown users, defaults to False.
        ignore_unknown_item: Ignore unknown items, defaults to False.
    """

    def __init__(
        self,
        metric_entries: list[MetricEntry],
        setting: Setting,
        metric_k: int,
        ignore_unknown_user: bool = False,
        ignore_unknown_item: bool = False,
        seed: int = 42,
    ) -> None:
        self.metric_entries = metric_entries
        self.setting = setting
        """Setting to evaluate the algorithms on."""
        self.metric_k = metric_k
        """Value of K for the metrics."""
        self.ignore_unknown_user = ignore_unknown_user
        """To ignore unknown users during evaluation."""
        self.ignore_unknown_item = ignore_unknown_item
        """To ignore unknown items during evaluation."""

        self.user_item_base = UserItemBaseStatus()
        self.seed = seed
        self._run_step = 0
        self._acc: MetricAccumulator
        self._current_timestamp: int

    def _get_evaluation_data(self) -> tuple[PredictionMatrix, PredictionMatrix, int]:
        """Get the evaluation data for the current step.

        Internal method to get the evaluation data for the current step. The
        evaluation data consists of the unlabeled data, ground truth data, and
        the current timestamp which will be returned as a tuple. The shapes
        are masked based through `user_item_base`. The unknown users in
        the ground truth data are also updated in `user_item_base`.

        Note:
            `_current_timestamp` is updated with the current timestamp.

        Returns:
            Tuple of unlabeled data, ground truth data, and current timestamp.

        Raises:
            EOWSettingError: If there is no more data to be processed.
        """
        try:
            split = self.setting.get_split_at(self._run_step)
            unlabeled_data = split.unlabeled
            ground_truth_data = split.ground_truth
            if split.t_window is None:
                raise ValueError("Timestamp of the current split cannot be None")
            self._current_timestamp = split.t_window

            unlabeled_data = PredictionMatrix.from_interaction_matrix(unlabeled_data)
            ground_truth_data = PredictionMatrix.from_interaction_matrix(ground_truth_data)
            self._run_step += 1
        except EOWSettingError:
            raise EOWSettingError("There is no more data to be processed, EOW reached")

        self.user_item_base.update_unknown_user_item_base(ground_truth_data)

        mask_shape = (self.user_item_base.known_shape[0], self.user_item_base.known_shape[1])
        if not self.ignore_unknown_user:
            mask_shape = (self.user_item_base.global_shape[0], mask_shape[1])

        unlabeled_data.mask_user_item_shape(
            shape=mask_shape
        )
        ground_truth_data.mask_user_item_shape(
            shape=mask_shape,
            drop_unknown_item=self.ignore_unknown_item,
            inherit_max_id=True,  # Ensures that shape of ground truth contains all user id that appears globally
        )
        # get the index of ground_truth_data._df
        if self.ignore_unknown_item:
            unlabeled_data._df = unlabeled_data._df.loc[ground_truth_data._df.index]
        return unlabeled_data, ground_truth_data, self._current_timestamp

    def _prediction_shape_handler(
        self, y_true: csr_matrix, y_pred: csr_matrix
    ) -> csr_matrix:
        """Handle shape difference of the prediction matrix.

        If there is a difference in the shape of the prediction matrix and the
        ground truth matrix, this function will handle the difference based on
        `ignore_unknown_user` and `ignore_unknown_item`.

        Args:
            X_true: Ground truth matrix.
            X_pred: Prediction matrix.
        """
        X_true_shape = y_true.shape
        if y_pred.shape != X_true_shape:
            logger.warning("Prediction matrix shape %s is different from ground truth matrix shape %s.", y_pred.shape, X_true_shape)
            # We cannot expect the algorithm to predict an unknown item, so we
            # only check user dimension
            if y_pred.shape[0] < X_true_shape[0] and not self.ignore_unknown_user:  # type: ignore
                raise ValueError(
                    "Prediction matrix shape, user dimension, is less than the ground truth matrix shape."
                )

            if not self.ignore_unknown_item:
                # prediction matrix would not contain unknown item ID
                # update the shape of the prediction matrix to include the ID
                y_pred = csr_matrix(
                    (y_pred.data, y_pred.indices, y_pred.indptr),
                    shape=(y_pred.shape[0], X_true_shape[1]),  # type: ignore
                )

            # shapes might not be the same in the case of dropping unknowns
            # from the ground truth data. We ensure that the same unknowns
            # are dropped from the predictions
            if self.ignore_unknown_user:
                y_pred = y_pred[: X_true_shape[0], :]  # type: ignore
            if self.ignore_unknown_item:
                y_pred = y_pred[:, : X_true_shape[1]]  # type: ignore

        return y_pred

    def metric_results(
        self,
        level: MetricLevelEnum | Literal["macro", "micro", "window", "user"] = MetricLevelEnum.MACRO,
        only_current_timestamp: None | bool = False,
        filter_timestamp: None | int = None,
        filter_algo: None | str = None,
    ) -> pd.DataFrame:
        """Results of the metrics computed.

        Computes the metrics of all algorithms based on the level specified and
        return the results in a pandas DataFrame. The results can be filtered
        based on the algorithm name and the current timestamp.

        Specifics
        ---------
        - User level: User level metrics computed across all timestamps.
        - Window level: Window level metrics computed across all timestamps. This can
            be viewed as a macro level metric in the context of a single window, where
            the scores of each user is averaged within the window.
        - Macro level: Macro level metrics computed for entire timeline. This
            score is computed by averaging the scores of all windows, treating each
            window equally.
        - Micro level: Micro level metrics computed for entire timeline. This
            score is computed by averaging the scores of all users, treating each
            user and the timestamp the user is in as unique contribution to the
            overall score.

        Args:
            level: Level of the metric to compute, defaults to "macro".
            only_current_timestamp: Filter only the current timestamp, defaults to False.
            filter_timestamp: Timestamp value to filter on, defaults to None.
                If both `only_current_timestamp` and `filter_timestamp` are provided,
                `filter_timestamp` will be used.
            filter_algo: Algorithm name to filter on, defaults to None.

        Returns:
            Dataframe representation of the metric.
        """
        if isinstance(level, str) and not MetricLevelEnum.has_value(level):
            raise ValueError("Invalid level specified")
        level = MetricLevelEnum(level)

        if only_current_timestamp and filter_timestamp:
            raise ValueError("Cannot specify both only_current_timestamp and filter_timestamp.")

        timestamp = None
        if only_current_timestamp:
            timestamp = self._current_timestamp

        if filter_timestamp:
            timestamp = filter_timestamp

        return self._acc.df_metric(filter_algo=filter_algo, filter_timestamp=timestamp, level=level)

    def restore(self) -> None:
        """Restore the generators before pickling.

        This method is used to restore the generators after loading the object
        from a pickle file.
        """
        self.setting.restore(self._run_step)
        logger.debug("Generators restored")

    def current_step(self) -> int:
        """Return the current step of the evaluator.

        Returns:
            Current step of the evaluator.
        """
        return self._run_step
