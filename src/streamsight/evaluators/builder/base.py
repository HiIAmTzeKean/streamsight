import logging
from abc import ABC, abstractmethod
from warnings import warn

from streamsight.registries import (
    METRIC_REGISTRY,
    MetricEntry,
)
from streamsight.settings import Setting
from streamsight.utils import arg_to_str
from ..base import EvaluatorBase


logger = logging.getLogger(__name__)


class Builder(ABC):
    """Base class for Builder objects.

    Provides methods to set specific values for the builder and enforce checks
    such that the builder can be constructed correctly and to avoid possible
    errors when the builder is executed.
    """

    def __init__(
        self,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize the Builder.

        Args:
            ignore_unknown_user: Ignore unknown user in the evaluation.
            ignore_unknown_item: Ignore unknown item in the evaluation.
            seed: Random seed for reproducibility.
        """
        self.metric_entries: dict[str, MetricEntry] = dict()
        """dict of metrics to evaluate algorithm on.
        Using dict instead of list for fast lookup"""
        self.setting: Setting
        """Setting to evaluate the algorithms on"""
        self.ignore_unknown_user = ignore_unknown_user
        """Ignore unknown user in the evaluation"""
        self.ignore_unknown_item = ignore_unknown_item
        """Ignore unknown item in the evaluation"""
        self.metric_k: int
        self.seed: int = seed

    def _check_setting_exist(self) -> bool:
        """Check if setting is already set.

        Returns:
            True if setting is set, False otherwise.
        """
        return not (not hasattr(self, "setting") or self.setting is None)

    def set_metric_K(self, K: int) -> None:
        """Set K value for all metrics.

        Args:
            K: K value to set for all metrics.
        """
        self.metric_k = K

    def add_metric(self, metric: str | type) -> None:
        """Add metric to evaluate algorithm on.

        Metric will be added to the metric_entries dict where it will later be
        converted to a list when the evaluator is constructed.

        Note:
            If K is not yet specified, the setting's top_K value will be used. This
            requires the setting to be set before adding the metric.

        Args:
            metric: Metric to evaluate algorithm on.

        Raises:
            ValueError: If metric is not found in METRIC_REGISTRY.
            RuntimeError: If setting is not set.
        """
        if not self._check_setting_exist():
            raise RuntimeError(
                "Setting has not been set. To ensure conformity, of the addition of"
                " other components please set the setting first. Call add_setting() method."
            )

        metric = arg_to_str(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} could not be resolved.")

        if not hasattr(self, "metric_k"):
            self.metric_k = self.setting.top_K
            warn(
                "K value not yet specified before setting metric, using setting's top_K value."
                " We recommend specifying K value for metric. If you want to change the K value,"
                " you can clear all metric entry and set the K value before adding metrics."
            )

        metric_name = f"{metric}_{self.metric_k}"
        if metric_name in self.metric_entries:
            logger.warning(f"Metric {metric_name} already exists. Skipping adding metric.")
            return

        self.metric_entries[metric_name] = MetricEntry(metric, self.metric_k)

    def add_setting(self, setting: Setting) -> None:
        """Add setting to the evaluator builder.

        Note:
            The setting should be set before adding metrics or algorithms
            to the evaluator.

        Args:
            setting: Setting to evaluate the algorithms on.

        Raises:
            ValueError: If setting is not of instance Setting.
        """
        if not isinstance(setting, Setting):
            raise ValueError(f"setting should be of type Setting, got {type(setting)}")
        if hasattr(self, "setting") and self.setting is not None:
            warn("Setting is already set. Continuing will overwrite the setting.")

        self.setting = setting

    def clear_metrics(self) -> None:
        """Clear all metrics from the builder."""
        self.metric_entries.clear()

    def _check_ready(self) -> None:
        """Check if the builder is ready to construct Evaluator.

        Raises:
            RuntimeError: If there are invalid configurations.
        """
        if not hasattr(self, "metric_k"):
            self.metric_k = self.setting.top_K
            warn(
                "K value not yet specified before setting metric, using setting's top_K value."
                " We recommend specifying K value for metric. If you want to change the K value,"
                " you can clear all metric entry and set the K value before adding metrics."
            )

        if len(self.metric_entries) == 0:
            raise RuntimeError("No metrics specified, can't construct Evaluator")

        # Check for settings #
        if self.setting is None:
            raise RuntimeError("No settings specified, can't construct Evaluator")
        if not self.setting.is_ready:
            raise RuntimeError(
                "Setting is not ready, can't construct Evaluator. "
                "Call split() on the setting first."
            )

    @abstractmethod
    def build(self) -> EvaluatorBase:
        """Build object.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError
