import logging
import random
import uuid
import warnings
from typing import Dict, List, Optional, Union
from uuid import UUID
from warnings import warn

from scipy.sparse import csr_matrix

from streamsight.algorithms import Algorithm
from streamsight.evaluators.accumulator import MetricAccumulator
from streamsight.evaluators.base import EvaluatorBase
from streamsight.evaluators.util import AlgorithmStatusWarning
from streamsight.matrix import InteractionMatrix
from streamsight.metrics import Metric
from streamsight.registries import (METRIC_REGISTRY, AlgorithmStateEnum,
                                    AlgorithmStatusEntry,
                                    AlgorithmStatusRegistry, MetricEntry)
from streamsight.settings import EOWSetting, Setting

logger = logging.getLogger(__name__)

warnings.simplefilter('always', AlgorithmStatusWarning)

class EvaluatorStreamer(EvaluatorBase):
    """Evaluation via streaming through API
    
    The diagram below shows the diagram of the streamer evaluator for the 
    sliding window setting. Instead of the pipeline, we allow the user to
    stream the data release to the algorithm. The data communication is shown
    between the evaluator and the algorithm. Note that while only 2 splits are
    shown here, the evaluator will continue to stream the data until the end
    of the setting where there are no more splits.
    
    .. image:: /_static/stream_scheme.png
        :align: center
        :scale: 40 %
        :alt: Streamer diagram
    
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

    :param metric_entries: List of metric entries
    :type metric_entries: List[MetricEntry]
    :param setting: Setting object
    :type setting: Setting
    :param metric_k: Number of top interactions to consider
    :type metric_k: int
    :param ignore_unknown_user: To ignore unknown users
    :type ignore_unknown_user: bool
    :param ignore_unknown_item: To ignore unknown items
    :type ignore_unknown_item: bool
    :param seed: Random seed for the evaluator
    :type seed: Optional[int]
    """
    def __init__(
        self,
        metric_entries: List[MetricEntry],
        setting: Setting,
        metric_k: int,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(
            metric_entries,
            setting,
            metric_k,
            ignore_unknown_user,
            ignore_unknown_item,
            seed
        )

        self.status_registry = AlgorithmStatusRegistry()

        self._unlabeled_data_cache: InteractionMatrix
        self._ground_truth_data_cache: InteractionMatrix
        self._training_data_cache: InteractionMatrix

        self.has_started = False
        self.rd = random.Random(self.seed)

    def start_stream(self):
        """Start the streaming process
        
        This method is called to start the streaming process. The method will
        prepare the evaluator for the streaming process. The method will reset
        the data generators, prepare the micro and macro accumulators, update
        the known user/item base, and cache the evaluation data.
        
        The method will set the internal state :attr:`has_started` to True. The
        method can be called anytime after the evaluator is instantiated. However,
        once the method is called, the evaluator cannot register any new algorithms.
        
        :raises ValueError: If the stream has already started
        """
        # ? allow programmer to register anytime
        if self.has_started:
            raise ValueError("Cannot start the stream again")

        self.has_started = True
        self.setting.reset_data_generators()

        logger.debug(f"Preparing evaluator for streaming")
        self._acc = MetricAccumulator()
        background_data = self.setting.background_data
        self.user_item_base._update_known_user_item_base(background_data)
        background_data.mask_shape(self.user_item_base.known_shape)
        self._training_data_cache = background_data

        self._cache_evaluation_data()

    def register_algorithm(
        self,
        algorithm: Optional[Algorithm] = None,
        algorithm_name: Optional[str] = None,
    ) -> UUID:
        """Register the algorithm with the evaluator

        This method is called to register the algorithm with the evaluator.
        The method will assign a unique identifier to the algorithm and store
        the algorithm in the registry. The method will raise a ValueError if
        the stream has already started.

        :param algorithm: The algorithm to be registered
        :type algorithm: Algorithm
        :raises ValueError: If the stream has already started
        :raises ValueError: If neither algorithm nor algorithm_name is provided
        :return: The unique identifier of the algorithm
        :rtype: UUID
        """
        if self.has_started:
            raise ValueError(
                "Cannot register algorithm after the stream has started"
            )

        if algorithm is None and algorithm_name is None:
            raise ValueError(
                "Either 'algorithm' or 'algorithm_name' must be provided"
            )

        if algorithm and hasattr(algorithm, "identifier"):
            name = algorithm.identifier
            algo_ptr = algorithm
        else:
            name = algorithm_name
            algo_ptr = None

        if name is None:
            raise ValueError(
                "No valid name provided for the algorithm, either the algorithm"
                " must implement an identifier property or a name must be provided"
            )

        # assign a unique identifier to the algorithm
        algo_id = uuid.UUID(int=self.rd.getrandbits(128), version=4)

        logger.info(
            f"Registering algorithm name {algorithm_name} with ID: {algo_id}"
        )

        # store the algorithm in the registry
        self.status_registry[algo_id] = AlgorithmStatusEntry(
            name=name,
            algo_id=algo_id,
            state=AlgorithmStateEnum.NEW,
            algo_ptr=algo_ptr,
        )
        logger.debug(f"Algorithm {algo_id} registered")
        return algo_id

    def get_algorithm_state(self, algo_id: UUID) -> AlgorithmStateEnum:
        """Get the state of the algorithm
        
        This method is called to get the state of the algorithm given the
        unique identifier of the algorithm. The method will return the state
        of the algorithm.

        :param algo_id: Unique identifier of the algorithm
        :type algo_id: UUID
        :return: The state of the algorithm
        :rtype: AlgorithmStateEnum
        """
        return self.status_registry[algo_id].state

    def get_all_algorithm_status(self) -> Dict[str, AlgorithmStateEnum]:
        """Get the status of all algorithms
        
        This method is called to get the status of all algorithms registered
        with the evaluator. The method will return a dictionary where the key
        is the name of the algorithm and the value is the state of the algorithm.

        :return: The status of all algorithms
        :rtype: Dict[str, AlgorithmStateEnum]
        """
        return self.status_registry.all_algo_states()

    def get_data(self, algo_id: UUID) -> InteractionMatrix:
        """Get training data for the algorithm
        
        Summary
        --------
        
        This method is called to get the training data for the algorithm. The
        training data is defined as either the background data or the incremental
        data. The training data is always released irrespective of the state of
        the algorithm.
        
        Specifics
        --------
        
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

        :param algo_id: Unique identifier of the algorithm
        :type algo_id: UUID
        :raises ValueError: If the stream has not started
        :return: The training data for the algorithm
        :rtype: InteractionMatrix
        """
        if not self.has_started:
            raise ValueError(f"call start_stream() before requesting data for algorithm {algo_id}")

        # check if we need to move to the next window
        if self.setting.is_sliding_window_setting and self.status_registry.is_all_predicted():
            self.user_item_base._reset_unknown_user_item_base()
            incremental_data = self.setting.next_incremental_data()
            self.user_item_base._update_known_user_item_base(incremental_data)
            incremental_data.mask_shape(self.user_item_base.known_shape)
            self._training_data_cache = incremental_data

            self._cache_evaluation_data()

        # check status of of algo
        status = self.status_registry[algo_id].state

        if status == AlgorithmStateEnum.COMPLETED:
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))

        elif status == AlgorithmStateEnum.NEW:
            self.status_registry.update(algo_id, AlgorithmStateEnum.READY, self._current_timestamp)

        elif status == AlgorithmStateEnum.READY and self.status_registry[algo_id].data_segment == self._current_timestamp:
            warn(AlgorithmStatusWarning(algo_id, status, "data_release"))

        elif status == AlgorithmStateEnum.PREDICTED and self.status_registry[algo_id].data_segment == self._current_timestamp:
            # if algo has predicted, check if current timestamp has not changed
            return_msg = f"Algorithm {algo_id} has already predicted for this data segment, please wait for all other algorithms to predict"
            warn(AlgorithmStatusWarning(algo_id, status, "not_all_predicted"))
            logger.info(return_msg)
            print(return_msg)

        else:
            # ? any other scenario that we have not accounted for
            self.status_registry.update(algo_id, AlgorithmStateEnum.READY, self._current_timestamp)

        # release data to the algorithm
        return self._training_data_cache

    def get_unlabeled_data(self, algo_id: UUID) -> Optional[InteractionMatrix]:
        """Get unlabeled data for the algorithm

        Summary
        --------
        
        This method is called to get the unlabeled data for the algorithm. The
        unlabeled data is the data that the algorithm will predict. It will
        contain `(user_id, -1)` pairs, where the value -1 indicates that the
        item is to be predicted.
        
        Specifics
        --------
        
        1. If the state is READY/PREDICTED, return the unlabeled data
        2. If the state is COMPLETED, raise warning that the algorithm has completed
        3. ALl other same, raise warning that the algorithm has not obtained data
           and should call :meth:`get_data` first.
        
        
        :param algo_id: Unique identifier of the algorithm
        :type algo_id: UUID
        :return: The unlabeled data for prediction
        :rtype: Optional[InteractionMatrix]
        """
        status = self.status_registry[algo_id].state
        if status in [AlgorithmStateEnum.READY, AlgorithmStateEnum.PREDICTED]:
            return self._unlabeled_data_cache

        if status == AlgorithmStateEnum.COMPLETED:
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))
            return
        warn(AlgorithmStatusWarning(algo_id, status, "unlabeled"))

    def submit_prediction(self, algo_id: UUID, X_pred: Union[csr_matrix, InteractionMatrix]) -> None:
        """Submit the prediction of the algorithm
        
        Summary
        --------
        
        This method is called to submit the prediction of the algorithm.
        There are a few checks that are done before the prediction is
        evaluated by calling :meth:`_evaluate`.
        
        Once the prediction is evaluated, the method will update the state
        of the algorithm to PREDICTED.
        
        The method will also check for each call if the current step of evaluation
        is the final one, if it is the final step, the method will update the
        state of the algorithm to COMPLETED.
        
        Specifics
        --------
        
        1. If state is READY, evaluate the prediction
        2. If state is NEW, algorithm has not obtained data, raise warning
        3. If state is PREDICTED, algorithm has already predicted, raise warning
        4. All other state, raise warning that the algorithm has completed
        

        :param algo_id: The unique identifier of the algorithm
        :type algo_id: UUID
        :param X_pred: The prediction of the algorithm
        :type X_pred: csr_matrix
        :raises ValueError: If X_pred is not an InteractionMatrix or csr_matrix
        """
        status = self.status_registry[algo_id].state

        if status == AlgorithmStateEnum.READY:
            try:
                X_pred = self._transform_prediction(X_pred)
            except ValueError as e:
                warn(f"Prediction failed for algorithm {algo_id} due to {e}")
                return
            # TODO check if all requested prediction made #86 bug
            self._evaluate(algo_id, X_pred)
            self.status_registry.update(algo_id, AlgorithmStateEnum.PREDICTED)

        elif status == AlgorithmStateEnum.NEW:
            warn(AlgorithmStatusWarning(algo_id, status, "predict"))

        elif status == AlgorithmStateEnum.PREDICTED:
            warn(AlgorithmStatusWarning(algo_id, status, "predict_complete"))

        else:
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))

        if self._run_step == self.setting.num_split:
            self.status_registry.update(algo_id, AlgorithmStateEnum.COMPLETED)
            logger.info(f"Finished streaming")
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))

    def _transform_prediction(self, X_pred: Union[csr_matrix, InteractionMatrix]) -> csr_matrix:
        """Transform the prediction matrix
        
        This method is called to transform the prediction matrix to a csr_matrix.
        The method will check if the prediction matrix is an InteractionMatrix
        and if the shape attribute is defined. If the shape attribute is not
        defined, the method will set the shape to the known shape of the user/item
        base.

        :param X_pred: The prediction matrix
        :type X_pred: Union[csr_matrix, InteractionMatrix]
        :raises ValueError: If X_pred is not an InteractionMatrix or csr_matrix
        :return: The prediction matrix as a csr_matrix
        :rtype: csr_matrix
        """
        if isinstance(X_pred, InteractionMatrix):
            # check if shape is defined
            if not hasattr(X_pred, "shape"):
                # prediction may be done on unknown users as well
                # we mask based on the larger shape
                X_pred.mask_shape(self.user_item_base.global_shape)
            # if there still exists ID outside the global shape, then the algorithm
            # is giving us ID that is not known to us, raise exception
            if X_pred.user_ids.difference(self.user_item_base.global_user_ids) or X_pred.item_ids.difference(self.user_item_base.global_item_ids):
                raise ValueError("Prediction matrix contains unknown user/item ids")
            X_pred = X_pred.binary_values
        elif isinstance(X_pred, csr_matrix):
            pass
        else:
            raise ValueError("X_pred must be either InteractionMatrix or csr_matrix")
        return X_pred

    def _cache_evaluation_data(self) -> None:
        """Cache the evaluation data for the current step.
        
        Summary
        --------
        This method will cache the evaluation data for the current step. The method
        will update the unknown user/item base, get the next unlabeled and ground
        truth data, and update the current timestamp.
        
        Specifics
        --------
        The method will update the unknown user/item base with the ground truth data.
        Next, mask the unlabeled and ground truth data with the known user/item
        base. The method will cache the unlabeled and ground truth data in the internal
        attributes :attr:`_unlabeled_data_cache` and :attr:`_ground_truth_data_cache`.
        The timestamp is cached in the internal attribute :attr:`_current_timestamp`.
        
        we use an internal attribute :attr:`_run_step` to keep track of the current
        step such that we can check if we have reached the last step.
        
        We assume that any method calling this method has already checked if the
        there is still data to be processed.
        """
        self._run_step += 1

        logger.debug(f"Caching evaluation data for step {self._run_step}")
        
        try:
            unlabeled_data, ground_truth_data, _ = self._get_evaluation_data()
        except EOWSetting as e:
            raise e

        self._unlabeled_data_cache = unlabeled_data
        self._ground_truth_data_cache = ground_truth_data
        
        logger.debug(f"Data cached for step {self._run_step} complete")

    def _evaluate(self, algo_id: UUID, X_pred: csr_matrix) -> None:
        """Evaluate the prediction
        
        Given the prediction and the algorithm ID, the method will evaluate the
        prediction using the metrics specified in the evaluator. The prediction
        of the algorithm is compared to the ground truth data currently cached.
        
        The evaluation results will be stored in the micro and macro accumulators
        which will later be used to calculate the final evaluation results.

        :param algo_id: The unique identifier of the algorithm
        :type algo_id: UUID
        :param X_pred: The prediction of the algorithm
        :type X_pred: csr_matrix
        """
        X_true = self._ground_truth_data_cache.get_users_n_first_interaction(self.metric_k)
        X_true = X_true.binary_values
        
        X_pred = self._prediction_shape_handler(X_true.shape, X_pred)
        algorithm_name = self.status_registry.get_algorithm_identifier(algo_id)

        # evaluate the prediction
        for metric_entry in self.metric_entries:
            metric_cls = METRIC_REGISTRY.get(metric_entry.name)
            if metric_entry.K is not None:
                metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=self._current_timestamp)
            else:
                metric:Metric = metric_cls(timestamp_limit=self._current_timestamp)
            metric.calculate(X_true, X_pred)
            self._acc.add(metric=metric, algorithm_name=algorithm_name)
        
        logger.debug(f"Prediction evaluated for algorithm {algo_id} complete")
