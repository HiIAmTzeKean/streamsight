from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np
from streamsight.evaluators.evaluator_pipeline import EvaluatorPipeline
from streamsight.evaluators.evaluator_stream import EvaluatorStreamer
from streamsight.registries import (
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    AlgorithmEntry,
    MetricEntry,
)
from streamsight.settings import Setting
from streamsight.utils import arg_to_str

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
        seed: Optional[int] = None
    ):
        self.metric_entries: Dict[str, MetricEntry] = dict()
        """Dict of metrics to evaluate algorithm on.
        Using Dict instead of List for fast lookup"""
        self.setting: Setting
        """Setting to evaluate the algorithms on"""
        self.ignore_unknown_user = ignore_unknown_user
        """Ignore unknown user in the evaluation"""
        self.ignore_unknown_item = ignore_unknown_item
        """Ignore unknown item in the evaluation"""
        self.metric_k: int
        if not seed:
            seed = 42
        self.seed: int = seed

    def _check_setting_exist(self):
        """Check if setting is already set.

        :raises RuntimeError: If setting has not been set
        """
        if not hasattr(self, "setting") or self.setting is None:
            raise RuntimeError("Setting has not been set. To ensure conformity, "
                               "of the addition of other components please set "
                               "the setting first. Call add_setting() method.")
        return True

    def set_metric_K(self, K: int) -> None:
        """Set K value for all metrics.

        :param K: K value to set for all metrics
        :type K: int
        """
        self.metric_k = K

    def add_metric(
        self, metric: Union[str, type]
    ) -> None:
        """Add metric to evaluate algorithm on.
        
        Metric will be added to the metric_entries dict where it will later be
        converted to a list when the evaluator is constructed.
        
        .. note::
            If K is not yet specified, the setting's top_K value will be used. This
            requires the setting to be set before adding the metric.

        :param metric: Metric to evaluate algorithm on
        :type metric: Union[str, type]
        :param K: Top K value to evaluate the prediction on, defaults to None
        :type K: Optional[int], optional
        :raises ValueError: If metric is not found in METRIC_REGISTRY
        :raises RuntimeError: If setting is not set
        """
        try:
            self._check_setting_exist()
        except RuntimeError:
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
            logger.warning(
                f"Metric {metric_name} already exists."
                " Skipping adding metric."
            )
            return

        self.metric_entries[metric_name] = MetricEntry(metric, self.metric_k)
        
    def add_setting(self, setting: Setting) -> None:
        """Add setting to the evaluator builder.
        
        .. note::
            The setting should be set before adding metrics or algorithms
            to the evaluator.

        :param setting: Setting to evaluate the algorithms on
        :type setting: Setting
        :raises ValueError: If setting is not of instance Setting
        """
        if not isinstance(setting, Setting):
            raise ValueError(
                f"setting should be of type Setting, got {type(setting)}"
            )
        if hasattr(self, "setting") and self.setting is not None:
            warn("Setting is already set. Continuing will overwrite the setting.")
        
        self.setting = setting
    
    def clear_metrics(self) -> None:
        """Clear all metrics from the builder."""
        self.metric_entries.clear()
        self.metric_k = None
    
    def _check_ready(self):
        """Check if the builder is ready to construct Evaluator.

        :raises RuntimeError: If there are invalid configurations
        """
        if not hasattr(self, "metric_k"):
            self.metric_k = self.setting.top_K
            warn(
                "K value not yet specified before setting metric, using setting's top_K value."
                " We recommend specifying K value for metric. If you want to change the K value,"
                " you can clear all metric entry and set the K value before adding metrics."
            )
        
        if len(self.metric_entries) == 0:
            raise RuntimeError(
                "No metrics specified, can't construct Evaluator"
            )

        # Check for settings #
        if self.setting is None:
            raise RuntimeError(
                "No settings specified, can't construct Evaluator"
            )
        if not self.setting.is_ready:
            raise RuntimeError(
                "Setting is not ready, can't construct Evaluator. "
                "Call split() on the setting first."
            )

    @abstractmethod
    def build(self):
        """Build object.

        :raises NotImplementedError: If the method is not implemented
        """
        raise NotImplementedError


class EvaluatorPipelineBuilder(Builder):
    """Builder to facilitate construction of evaluator.
    Provides methods to set specific values for the evaluator and enforce checks
    such that the evaluator can be constructed correctly and to avoid possible
    errors when the evaluator is executed.
    
    :param ignore_unknown_user: Ignore unknown user in the evaluation, defaults to True
    :type ignore_unknown_user: bool, optional
    :param ignore_unknown_item: Ignore unknown item in the evaluation, defaults to True
    :type ignore_unknown_item: bool, optional
    """
    def __init__(
        self,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__(ignore_unknown_user, ignore_unknown_item, seed)
        self.algorithm_entries: List[AlgorithmEntry] = []
        """List of algorithms to evaluate"""

    def add_algorithm(
        self,
        algorithm: Union[str, type],
        params: Optional[Dict[str, int]] = None,
    ):
        """Add algorithm to evaluate.

        Adding algorithm to evaluate on. The algorithm can be added by specifying the class type
        or by specifying the class name as a string.

        :param algorithm: Algorithm to evaluate
        :type algorithm: Union[str, type]
        :param params: Parameter for the algorithm, defaults to None
        :type params: Optional[Dict[str, int]], optional
        :raises ValueError: If algorithm is not found in ALGORITHM_REGISTRY
        """
        try:
            self._check_setting_exist()
        except RuntimeError:
            raise RuntimeError(
                "Setting has not been set. To ensure conformity, of the addition of"
                " other components please set the setting first. Call add_setting() method."
            )
        
        algorithm = arg_to_str(algorithm)

        #? additional check for K value mismatch with setting
        
        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm {algorithm} could not be resolved.")

        self.algorithm_entries.append(AlgorithmEntry(algorithm, params or {}))

    def _check_ready(self):
        """Check if the builder is ready to construct Evaluator.

        :raises RuntimeError: If there are invalid configurations
        """
        super()._check_ready()
        
        if len(self.algorithm_entries) == 0:
            raise RuntimeError(
                "No algorithms specified, can't construct Evaluator"
            )

        for algo in self.algorithm_entries:
            if (
                algo.params is not None
                and "K" in algo.params
                and algo.params["K"] < self.setting.top_K
            ):
                warn(
                    f"Algorithm {algo.name} has K={algo.params['K']} but setting"
                    f" is configured top_K={self.setting.top_K}. The number of predictions"
                    " returned by the model is less than the K value to evaluate the predictions"
                    " on. This may distort the metric value."
                )

    def build(self) -> EvaluatorPipeline:
        """Build Evaluator object.

        :raises RuntimeError: If no metrics, algorithms or settings are specified
        :return: Evaluator object
        :rtype: Evaluator
        """
        self._check_ready()
        return EvaluatorPipeline(
            algorithm_entries=self.algorithm_entries,
            metric_entries=list(self.metric_entries.values()),
            setting=self.setting,
            metric_k=self.metric_k,
            ignore_unknown_user=self.ignore_unknown_user,
            ignore_unknown_item=self.ignore_unknown_item,
            seed=self.seed
        )


class EvaluatorStreamerBuilder(Builder):
    """Builder to facilitate construction of evaluator.
    Provides methods to set specific values for the evaluator and enforce checks
    such that the evaluator can be constructed correctly and to avoid possible
    errors when the evaluator is executed.
    
    :param ignore_unknown_user: Ignore unknown user in the evaluation, defaults to True
    :type ignore_unknown_user: bool, optional
    :param ignore_unknown_item: Ignore unknown item in the evaluation, defaults to True
    :type ignore_unknown_item: bool, optional
    """
    def __init__(
        self,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__(ignore_unknown_user, ignore_unknown_item, seed)

    def build(self) -> EvaluatorStreamer:
        """Build Evaluator object.

        :raises RuntimeError: If no metrics, algorithms or settings are specified
        :return: Evaluator object
        :rtype: Evaluator
        """
        self._check_ready()
        return EvaluatorStreamer(
            metric_entries=list(self.metric_entries.values()),
            setting=self.setting,
            metric_k=self.metric_k,
            ignore_unknown_user=self.ignore_unknown_user,
            ignore_unknown_item=self.ignore_unknown_item,
            seed=self.seed
        )
