import logging

from streamsight.evaluators.evaluator_stream import EvaluatorStreamer
from .base import Builder


logger = logging.getLogger(__name__)


class EvaluatorStreamerBuilder(Builder):
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
        """Initialize the EvaluatorStreamerBuilder.

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

    def build(self) -> EvaluatorStreamer:
        """Build Evaluator object.

        Raises:
            RuntimeError: If no metrics, algorithms or settings are specified.

        Returns:
            EvaluatorStreamer: The built evaluator object.
        """
        self._check_ready()
        return EvaluatorStreamer(
            metric_entries=list(self.metric_entries.values()),
            setting=self.setting,
            metric_k=self.metric_k,
            ignore_unknown_user=self.ignore_unknown_user,
            ignore_unknown_item=self.ignore_unknown_item,
            seed=self.seed,
        )
