import logging
from enum import Enum
from uuid import UUID

from scipy.sparse import csr_matrix

from streamsight.algorithms import Algorithm
from streamsight.matrix import InteractionMatrix, PredictionMatrix
from streamsight.metrics import Metric
from streamsight.registries import (
    METRIC_REGISTRY,
    MetricEntry,
)
from streamsight.settings import EOWSettingError, Setting
from .accumulator import MetricAccumulator
from .base import EvaluatorBase
from .state_management import AlgorithmStateManager
from .strategy import EvaluationStrategy, SlidingWindowStrategy
from .warning import AlgorithmStateEnum


class EvaluatorState(Enum):
    """Evaluator lifecycle states"""

    INITIALIZED = "initialized"
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


logger = logging.getLogger(__name__)


class EvaluatorStreamer(EvaluatorBase):
    """Evaluation via streaming through API.

    The diagram below shows the diagram of the streamer evaluator for the
    sliding window setting. Instead of the pipeline, we allow the user to
    stream the data release to the algorithm. The data communication is shown
    between the evaluator and the algorithm. Note that while only 2 splits are
    shown here, the evaluator will continue to stream the data until the end
    of the setting where there are no more splits.

    ![stream scheme](../../../assets/_static/stream_scheme.png)

    This class exposes a few of the core API that allows the user to stream
    the evaluation process. The following API are exposed:

    1. :meth:`register_algorithm`
    2. :meth:`start_stream`
    3. :meth:`get_unlabeled_data`
    4. :meth:`submit_prediction`

    The programmer can take a look at the specific method for more details
    on the implementation of the API. The methods are designed with the
    methodological approach that the algorithm is decoupled from the
    the evaluating platform. And thus, the evaluator will only provide
    the necessary data to the algorithm and evaluate the prediction.

    Args:
        metric_entries: list of metric entries.
        setting: Setting object.
        metric_k: Number of top interactions to consider.
        ignore_unknown_user: To ignore unknown users.
        ignore_unknown_item: To ignore unknown items.
        seed: Random seed for the evaluator.
    """

    def __init__(
        self,
        metric_entries: list[MetricEntry],
        setting: Setting,
        metric_k: int,
        ignore_unknown_user: bool = False,
        ignore_unknown_item: bool = False,
        seed: int = 42,
        strategy: None | EvaluationStrategy = None,
    ) -> None:
        super().__init__(
            metric_entries,
            setting,
            metric_k,
            ignore_unknown_user,
            ignore_unknown_item,
            seed,
        )
        self._algo_state_mgr = AlgorithmStateManager()
        self._unlabeled_data_cache: PredictionMatrix
        self._ground_truth_data_cache: PredictionMatrix
        self._training_data_cache: PredictionMatrix

        # Evaluator state management
        self._state = EvaluatorState.INITIALIZED

        # Evaluation strategy
        self._strategy = strategy or SlidingWindowStrategy()

    @property
    def state(self) -> EvaluatorState:
        return self._state

    def _assert_state(
        self,
        expected: EvaluatorState,
        error_msg: str
    ) -> None:
        """Assert evaluator is in expected state"""
        if self._state != expected:
            raise RuntimeError(
                f"{error_msg} (Current state: {self._state.value})"
            )

    def _transition_state(
        self,
        new_state: EvaluatorState,
        allow_from: list[EvaluatorState]
    ) -> None:
        """Guard state transitions explicitly"""
        if self._state not in allow_from:
            raise ValueError(
                f"Cannot transition from {self._state} to {new_state}. "
                f"Allowed from: {allow_from}"
            )
        self._state = new_state
        logger.info(f"Evaluator transitioned to {new_state}")

    def _cache_evaluation_data(self) -> None:
        """Cache the evaluation data for the current step.

        Summary
        -------
        This method will cache the evaluation data for the current step. The method
        will update the unknown user/item base, get the next unlabeled and ground
        truth data, and update the current timestamp.

        Specifics
        ---------
        The method will update the unknown user/item base with the ground truth data.
        Next, mask the unlabeled and ground truth data with the known user/item
        base. The method will cache the unlabeled and ground truth data in the internal
        attributes :attr:`_unlabeled_data_cache` and :attr:`_ground_truth_data_cache`.
        The timestamp is cached in the internal attribute :attr:`_current_timestamp`.

        We use an internal attribute :attr:`_run_step` to keep track of the current
        step such that we can check if we have reached the last step.

        We assume that any method calling this method has already checked if the
        there is still data to be processed.
        """

        logger.debug(f"Caching evaluation data for step {self._run_step}")
        try:
            self._unlabeled_data_cache, self._ground_truth_data_cache, _ = self._get_evaluation_data()
        except EOWSettingError as e:
            raise e
        logger.debug(f"Data cached for step {self._run_step} complete")

    def start_stream(self) -> None:
        """Start the streaming process.

        This method is called to start the streaming process. `start_stream` will
        prepare the evaluator for the streaming process. `start_stream` will reset
        data streamers, prepare the micro and macro accumulators, update
        the known user/item base, and cache data.

        The method will set the internal state to be be started. The
        method can be called anytime after the evaluator is instantiated.

        Warning:
            Once `start_stream` is called, the evaluator cannot register any new algorithms.

        Raises:
            ValueError: If the stream has already started.
        """
        self.setting.restore()

        logger.debug("Preparing evaluator for streaming")
        self._acc = MetricAccumulator()
        training_data = self.setting.background_data
        # Convert to PredictionMatrix since it's a subclass of InteractionMatrix
        training_data = PredictionMatrix.from_interaction_matrix(training_data)

        self.user_item_base.update_known_user_item_base(training_data)
        training_data.mask_user_item_shape(self.user_item_base.known_shape)
        self._training_data_cache = training_data
        self._cache_evaluation_data()
        self._algo_state_mgr.set_all_ready(data_segment=self._current_timestamp)
        logger.debug("Evaluator is ready for streaming")
        # TODO: allow programmer to register anytime
        self._transition_state(
            EvaluatorState.STARTED,
            allow_from=[EvaluatorState.INITIALIZED]
        )

    def register_algorithm(
        self,
        algorithm: None | Algorithm = None,
        algorithm_name: None | str = None,
    ) -> UUID:
        """Register the algorithm with the evaluator.

        This method is called to register the algorithm with the evaluator.
        The method will assign a unique identifier to the algorithm and store
        the algorithm in the registry. The method will raise a ValueError if
        the stream has already started.
        """
        self._assert_state(
            EvaluatorState.INITIALIZED,
            "Cannot register algorithms after stream started"
        )
        algo_id = self._algo_state_mgr.register(name=algorithm_name, algo_ptr=algorithm)
        logger.debug(f"Algorithm {algo_id} registered")
        return algo_id

    def get_algorithm_state(self, algo_id: UUID) -> AlgorithmStateEnum:
        """Get the state of the algorithm.

        This method is called to get the state of the algorithm given the
        unique identifier of the algorithm. The method will return the state
        of the algorithm.

        Args:
            algo_id: Unique identifier of the algorithm.

        Returns:
            The state of the algorithm.
        """
        return self._algo_state_mgr[algo_id].state

    def get_all_algorithm_status(self) -> dict[str, AlgorithmStateEnum]:
        """Get the status of all algorithms.

        This method is called to get the status of all algorithms registered
        with the evaluator. The method will return a dictionary where the key
        is the name of the algorithm and the value is the state of the algorithm.

        Returns:
            The status of all algorithms.
        """
        return self._algo_state_mgr.all_algo_states()

    def load_next_window(self) -> None:
        self.user_item_base.reset_unknown_user_item_base()
        incremental_data = self.setting.get_split_at(self._run_step).incremental
        if incremental_data is None:
            raise EOWSettingError("No more data to stream")
        # Convert to PredictionMatrix since it's a subclass of InteractionMatrix
        incremental_data = PredictionMatrix.from_interaction_matrix(incremental_data)

        self.user_item_base.update_known_user_item_base(incremental_data)
        incremental_data.mask_user_item_shape(self.user_item_base.known_shape)
        self._training_data_cache = incremental_data
        self._cache_evaluation_data()
        self._algo_state_mgr.set_all_ready(data_segment=self._current_timestamp)

    def get_training_data(self, algo_id: UUID) -> InteractionMatrix:
        """Get training data for the algorithm.

        Summary
        -------

        This method is called to get the training data for the algorithm. The
        training data is defined as either the background data or the incremental
        data. The training data is always released irrespective of the state of
        the algorithm.

        Specifics
        ---------

        1. If the state is COMPLETED, raise warning that the algorithm has completed
        2. If the state is NEW, release training data to the algorithm
        3. If the state is READY and the data segment is the same, raise warning
           that the algorithm has already obtained data
        4. If the state is PREDICTED and the data segment is the same, inform
           the algorithm that it has already predicted and should wait for other
           algorithms to predict
        5. This will occur when :attr:`_current_timestamp` has changed, which causes
           scenario 2 to not be caught. In this case, the algorithm is requesting
           the next window of data. Thus, this is a valid data call and the status
           will be updated to READY.

        Args:
            algo_id: Unique identifier of the algorithm.

        Raises:
            ValueError: If the stream has not started.

        Returns:
            The training data for the algorithm.
        """
        self._assert_state(EvaluatorState.STARTED, "Call start_stream() first")

        logger.debug(f"Getting data for algorithm {algo_id}")

        if self._strategy.should_advance_window(
            algo_state_mgr=self._algo_state_mgr,
            current_step=self._run_step,
            total_steps=self.setting.num_split,
        ):
            try:
                self.load_next_window()
            except EOWSettingError:
                self._transition_state(
                    EvaluatorState.COMPLETED,
                    allow_from=[EvaluatorState.STARTED, EvaluatorState.IN_PROGRESS]
                )
                raise RuntimeError("End of evaluation window reached")

        can_request, reason = self._algo_state_mgr.can_request_training_data(algo_id)
        if not can_request:
            raise PermissionError(f"Cannot request data: {reason}")
        # TODO handle case when algo is ready after submitting prediction, but current timestamp has not changed, meaning algo is requesting same data again
        self._algo_state_mgr.transition(
            algo_id,
            AlgorithmStateEnum.RUNNING,
            data_segment=self._current_timestamp,
        )

        self._evaluator_state = EvaluatorState.IN_PROGRESS
        # release data to the algorithm
        return self._training_data_cache

    def get_unlabeled_data(self, algo_id: UUID) -> PredictionMatrix:
        """Get unlabeled data for the algorithm.

        This method is called to get the unlabeled data for the algorithm. The
        unlabeled data is the data that the algorithm will predict. It will
        contain `(user_id, -1)` pairs, where the value -1 indicates that the
        item is to be predicted.
        """
        logger.debug(f"Getting unlabeled data for algorithm {algo_id}")
        can_submit, reason = self._algo_state_mgr.can_request_unlabeled_data(algo_id)
        if not can_submit:
            raise PermissionError(f"Cannot get unlabeled data: {reason}")
        return self._unlabeled_data_cache

    def submit_prediction(self, algo_id: UUID, X_pred: csr_matrix) -> None:
        """Submit the prediction of the algorithm.

        This method is called to submit the prediction of the algorithm.
        There are a few checks that are done before the prediction is
        evaluated by calling :meth:`_evaluate_algo_pred`.

        Once the prediction is evaluated, the method will update the state
        of the algorithm to PREDICTED.
        """
        logger.debug(f"Submitting prediction for algorithm {algo_id}")
        can_submit, reason = self._algo_state_mgr.can_submit_prediction(algo_id)
        if not can_submit:
            raise PermissionError(f"Cannot submit prediction: {reason}")

        self._evaluate_algo_pred(algo_id=algo_id, y_pred=X_pred)
        self._algo_state_mgr.transition(
            algo_id,
            AlgorithmStateEnum.PREDICTED,
        )

    def _evaluate_algo_pred(self, algo_id: UUID, y_pred: csr_matrix) -> None:
        """Evaluate the prediction for algorithm.

        Given the prediction and the algorithm ID, the method will evaluate the
        prediction using the metrics specified in the evaluator. The prediction
        of the algorithm is compared to the ground truth data currently cached.

        The evaluation results will be stored in the micro and macro accumulators
        which will later be used to calculate the final evaluation results.

        Args:
            algo_id: The unique identifier of the algorithm.
            y_pred: The prediction of the algorithm.
        """
        # get top k ground truth interactions
        y_true = self._ground_truth_data_cache
        # y_true = self._ground_truth_data_cache.get_users_n_first_interaction(self.metric_k)
        y_true = y_true.item_interaction_sequence_matrix

        y_pred = self._prediction_shape_handler(y_true, y_pred)
        algorithm_name = self._algo_state_mgr.get_algorithm_identifier(algo_id)

        # evaluate the prediction
        for metric_entry in self.metric_entries:
            metric_cls = METRIC_REGISTRY.get(metric_entry.name)
            if metric_entry.K is not None:
                metric: Metric = metric_cls(K=metric_entry.K, timestamp_limit=self._current_timestamp)
            else:
                metric: Metric = metric_cls(timestamp_limit=self._current_timestamp)
            metric.calculate(y_true, y_pred)
            self._acc.add(metric=metric, algorithm_name=algorithm_name)

        logger.debug(f"Prediction evaluated for algorithm {algo_id} complete")
