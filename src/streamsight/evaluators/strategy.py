from abc import ABC, abstractmethod

from .state_management import AlgorithmStateManager


class EvaluationStrategy(ABC):
    """Abstract strategy for different evaluation modes"""

    @abstractmethod
    def should_advance_window(self, algo_state_mgr: AlgorithmStateManager, current_step: int, total_steps: int) -> bool:
        """Determine if should advance to next window"""
        pass


class SlidingWindowStrategy(EvaluationStrategy):
    """Strategy for sliding window evaluation"""

    def should_advance_window(self, algo_state_mgr: AlgorithmStateManager, current_step: int, total_steps: int) -> bool:
        """Advance only when all algorithms predicted"""
        return (
            algo_state_mgr.is_all_predicted()
            and algo_state_mgr.is_all_same_data_segment()
            and current_step < total_steps
        )


class SingleTimePointStrategy(EvaluationStrategy):
    """Strategy for sliding window evaluation"""

    def should_advance_window(self, algo_state_mgr: AlgorithmStateManager, current_step: int, total_steps: int) -> bool:
        """Advance only when all algorithms predicted"""
        return algo_state_mgr.is_all_predicted() and current_step < total_steps
