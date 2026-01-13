import logging
import uuid
from warnings import warn

from streamsight.algorithms import Algorithm
from streamsight.evaluators.evaluator_pipeline import EvaluatorPipeline
from ..state_management import AlgorithmStateManager
from .base import Builder


logger = logging.getLogger(__name__)


class EvaluatorPipelineBuilder(Builder):
    """Builder to facilitate construction of evaluator.

    Provides methods to set specific values for the evaluator and enforce checks
    such that the evaluator can be constructed correctly and to avoid possible
    errors when the evaluator is executed.
    """

    def __init__(
        self,
        ignore_unknown_user: bool = False,
        ignore_unknown_item: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize the EvaluatorPipelineBuilder.

        Args:
            ignore_unknown_user: Ignore unknown user in the evaluation.
            ignore_unknown_item: Ignore unknown item in the evaluation.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            ignore_unknown_user=ignore_unknown_user,
            ignore_unknown_item=ignore_unknown_item,
            seed=seed,
        )
        self.algo_state_mgr = AlgorithmStateManager()

    def add_algorithm(
        self,
        algorithm: type[Algorithm],
        params: dict[str, int] = {},
        algo_uuid: None | uuid.UUID = None,
    ) -> None:
        """Add algorithm to evaluate.

        Adding algorithm to evaluate on. The algorithm can be added by specifying the class type
        or by specifying the class name as a string.

        Args:
            algorithm: Algorithm to evaluate.
            params: Parameter for the algorithm.

        Raises:
            ValueError: If algorithm is not found in ALGORITHM_REGISTRY.
        """
        if not self._check_setting_exist():
            raise RuntimeError(
                "Setting has not been set. To ensure conformity, of the addition of"
                " other components please set the setting first. Call add_setting() method."
            )

        self.algo_state_mgr.register(algo_ptr=algorithm, params=params, algo_uuid=algo_uuid)

    def _check_ready(self) -> None:
        """Check if the builder is ready to construct Evaluator.

        Raises:
            RuntimeError: If there are invalid configurations.
        """
        super()._check_ready()

        if len(self.algo_state_mgr) == 0:
            raise RuntimeError("No algorithms specified, can't construct Evaluator")

        for algo_state in self.algo_state_mgr.values():
            if (
                algo_state.params is not None
                and "K" in algo_state.params
                and algo_state.params["K"] < self.setting.top_K
            ):
                warn(
                    f"Algorithm {algo_state.name} has K={algo_state.params['K']} but setting"
                    f" is configured top_K={self.setting.top_K}. The number of predictions"
                    " returned by the model is less than the K value to evaluate the predictions"
                    " on. This may distort the metric value."
                )

    def build(self) -> EvaluatorPipeline:
        """Build Evaluator object.

        Raises:
            RuntimeError: If no metrics, algorithms or settings are specified.

        Returns:
            EvaluatorPipeline: The built evaluator object.
        """
        self._check_ready()
        return EvaluatorPipeline(
            # algorithm_entries=self.algorithm_entries,
            algo_state_mgr=self.algo_state_mgr,
            metric_entries=list(self.metric_entries.values()),
            setting=self.setting,
            metric_k=self.metric_k,
            ignore_unknown_user=self.ignore_unknown_user,
            ignore_unknown_item=self.ignore_unknown_item,
            seed=self.seed,
        )
