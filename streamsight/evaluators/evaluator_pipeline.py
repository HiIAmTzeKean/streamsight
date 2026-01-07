import logging
import warnings
from typing import List, Optional
from warnings import warn

from tqdm import tqdm

from streamsight.algorithms import Algorithm
from streamsight.evaluators.accumulator import MetricAccumulator
from streamsight.evaluators.base import EvaluatorBase
from streamsight.metrics import Metric
from streamsight.registries import (ALGORITHM_REGISTRY, METRIC_REGISTRY,
                                    AlgorithmEntry, MetricEntry)
from streamsight.settings import EOWSetting, Setting

logger = logging.getLogger(__name__)

class EvaluatorPipeline(EvaluatorBase):
    """Evaluation via pipeline
    
    The diagram below shows the diagram of the pipeline evaluator for the 
    sliding window setting. If the setting is a single time point setting, the
    evaluator will only run phase 1 and 2. This is the same as setting the sliding
    window setting to only having 1 split.
    
    .. image:: /_static/pipeline_scheme.png
        :align: center
        :scale: 40 %
        :alt: Pipeline diagram
    
    The pipeline is responsible for evaluating algorithms with metrics and evaluates
    the algorithms in 3 phases:
    
    1. Pre-phase
    2. Evaluation phase
    3. Data release phase
    
    In Setting 3 (single time point split), the evaluator will only run phase 1 and 2.
    In Setting 1 (sliding window setting), the evaluator will run all 3 phases, with
    phase 2 and 3 repeated for each window/split. This can be seen in the diagram
    above.
    
    :param algorithm_entries: List of algorithm entries
    :type algorithm_entries: List[AlgorithmEntry]
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
        algorithm_entries: List[AlgorithmEntry],
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

        self.algorithm_entries = algorithm_entries
        self.algorithm: List[Algorithm]

        # internal state
        self._current_timestamp: int

    def _instantiate_algorithm(self):
        """Instantiate the algorithms from the algorithm entries.
        
        This method instantiates the algorithms and stores them in :attr:`algorithm`.
        Each time this method is called, the algorithms are re-instantiated.
        """
        self.algorithm = []
        for algorithm_entry in self.algorithm_entries:
            params = algorithm_entry.params
            self.algorithm.append(ALGORITHM_REGISTRY.get(algorithm_entry.name)(**params))

    def _ready_algo(self):
        """Train the algorithms with the background data.
        
        This method should be called after :meth:`_instantiate_algorithm()`. The
        algorithms are trained with the background data, and the set of known
        user/item is updated.

        :raises ValueError: If algorithm is not instantiated
        """
        if not hasattr(self, "algorithm"):
            raise ValueError("Algorithm not instantiated")
        background_data = self.setting.background_data
        self.user_item_base._update_known_user_item_base(background_data)
        # TODO timeline is not respected, can use flag to indicate a override the known user and item
        background_data.mask_shape(self.user_item_base.known_shape)

        for algo in self.algorithm:
            algo.fit(background_data)

    def _ready_evaluator(self):
        """Pre-phase of the evaluator. (Phase 1)
        
        Summary
        -------
        
        This method prepares the evaluator for the evaluation process.
        It instantiates the algorithm, trains the algorithm with the background data,
        instantiates the metric accumulator, and prepares the data generators.
        The next phase of the evaluator is the evaluation phase (Phase 2).
        
        Specifics
        ---------
        
        The evaluator is prepared by:
        1. Instantiating the each algorithm from the algorithm entries
        2. For each algorithm, train the algorithm with the background data from
           :attr:`setting`
        3. Instantiate the metric accumulator for micro and macro metrics
        4. Create an entry for each metric in the macro metric accumulator
        5. Prepare the data generators for the setting
        """
        logger.info("Phase 1: Preparing the evaluator...")
        self._instantiate_algorithm()
        self._ready_algo()
        logger.debug(f"Algorithms trained with background data...")

        self._acc = MetricAccumulator()
        logger.debug(f"Metric accumulator instantiated...")

        self.setting.reset_data_generators()
        logger.debug(f"Setting data generators ready...")

    def _evaluate_step(self):
        """Evaluate performance of the algorithms. (Phase 2)
        
        Summary
        -------
        
        This method evaluates the performance of the algorithms with the metrics.
        It takes the unlabeled data, predicts the interaction, and evaluates the
        performance with the ground truth data.
        
        Specifics
        ---------
        
        The evaluation is done by:
        1. Get the next unlabeled data and ground truth data from the setting
        2. Get the current timestamp from the setting
        3. Update the unknown user/item base with the ground truth data
        4. Mask the unlabeled data to the known user/item base shape
        5. Mask the ground truth data based on the provided flags :attr:`ignore_unknown_user`
           and :attr:`ignore_unknown_item`
        6. Predict the interaction with the algorithms
        7. Check the shape of the prediction matrix
        8. Store the results in the micro metric accumulator
        9. Cache the results in the macro metric accumulator
        10. Repeat step 6 for each algorithm
        
        :raises EOWSetting: If there is no more data to be processed
        """
        logger.info("Phase 2: Evaluating the algorithms...")
        try:
            unlabeled_data, ground_truth_data, current_timestamp = self._get_evaluation_data()
        except EOWSetting as e:
            raise e
        
        # get the top k interaction per user
        X_true = ground_truth_data.get_users_n_first_interaction(self.metric_k)
        X_true = X_true.binary_values
        for algo in self.algorithm:
            X_pred = algo.predict(unlabeled_data)
            X_pred = self._prediction_shape_handler(X_true.shape, X_pred)

            for metric_entry in self.metric_entries:
                metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=current_timestamp)
                metric.calculate(X_true, X_pred)
                self._acc.add(metric=metric, algorithm_name=algo.identifier)

    def _data_release_step(self):
        """Data release phase. (Phase 3)
        
        This method releases the data from the evaluator. This method should only
        be called when the setting is a sliding window setting.
        
        The data is released by updating the known user/item base with the
        incremental data. After updating the known user/item base, the incremental
        data is masked to the known user/item base shape. The algorithms are then
        trained with the incremental data.
        
        .. note::
            Previously unknown user/item base is reset after the data release.
            Since these unknown user/item base should be within the incremental
            data released.
        """
        if not self.setting.is_sliding_window_setting:
            return
        logger.info("Phase 3: Releasing the data...")

        incremental_data = self.setting.next_incremental_data()
        self.user_item_base._reset_unknown_user_item_base()
        self.user_item_base._update_known_user_item_base(incremental_data)
        incremental_data.mask_shape(self.user_item_base.known_shape)

        for algo in self.algorithm:
            algo.fit(incremental_data)

    def run_step(self, reset=False):
        """Run a single step of the evaluator.
        
        This method runs a single step of the evaluator. The evaluator is split
        into 3 phases. In the first run, all 3 phases are run. In the subsequent
        runs, only the evaluation and data release phase are run. The method
        will run all steps until the number of splits is reached. To rerun the
        evaluation again, call with `reset=True`.

        :param reset: To reset the evaluation step , defaults to False
        :type reset: bool, optional
        """
        if reset:
            self._run_step = 0

        self._run_step += 1
        if self._run_step == 1:
            logger.info(f"There is a total of {self.setting.num_split} steps."
                        f" Running step {self._run_step}")
            self._ready_evaluator()
            logger.info(f"Evaluator ready...")
            self._evaluate_step()
            self._data_release_step()
            return

        if self._run_step > self.setting.num_split:
            logger.info(f"Finished running all steps, call `run_step(reset=True)` to run the evaluation again")
            warn(f"Running this method again will not have any effect.")
            return
        logger.info(f"Running step {self._run_step}")
        self._evaluate_step()
        self._data_release_step()

    def run_steps(self, num_steps:int):
        """Run multiple steps of the evaluator.
        
        Effectively runs :meth:`run_step` method :param:`num_steps` times. Call
        this method to run multiple steps of the evaluator at once.

        :param num_steps: Number of steps to run
        :type num_steps: int
        """
        for _ in tqdm(range(num_steps)):
            self.run_step()      

    def run(self):
        """Run the evaluator across all steps and splits
        
        Runs all 3 phases across all splits (if there are multiple splits).
        This method should be called when the programmer wants to step through
        all phases and splits to arrive to the metrics computed. An alternative
        to running through all splits is to call :meth:`run_step` method which runs
        only one step at a time.
        """
        self._ready_evaluator()

        for _ in tqdm(range(self.setting.num_split)):
            self._evaluate_step()
            self._data_release_step()
