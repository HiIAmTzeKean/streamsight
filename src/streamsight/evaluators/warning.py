from uuid import UUID

from .state_management import AlgorithmStateEnum


class AlgorithmStatusWarning(UserWarning):
    """Warning for algorithm status issues.

    This warning is raised when an algorithm is in an inappropriate state
    for a requested operation.
    """

    def __init__(self, algo_id: UUID, status: AlgorithmStateEnum, phase: str) -> None:
        """Initialize the warning with algorithm details.

        Args:
            algo_id (UUID): The algorithm's unique identifier.
            status (AlgorithmStateEnum): The current status of the algorithm.
            phase (str): The phase where the issue occurred.
        """
        self.algo_id = algo_id
        self.status = status
        if phase == "data_release":
            super().__init__(
                f"Algorithm:{algo_id} current status is {status}. "
                "Algorithm has already requested for data. "
                "Returning the same data again."
            )
        elif phase == "unlabeled":
            super().__init__(
                f"Algorithm:{algo_id} not ready to get unlabeled data, "
                f"current status is {status}. Call get_training_data() first."
            )
        elif phase == "predict_complete":
            super().__init__(
                f"Algorithm:{algo_id} current status is {status}. "
                "Algorithm already submitted prediction"
            )
        elif phase == "predict":
            super().__init__(
                f"Algorithm:{algo_id} current status is {status}. "
                "Algorithm should request for unlabeled data first."
            )
        elif phase == "complete":
            super().__init__(
                f"Algorithm:{algo_id} current status is {status}. "
                "Algorithm has completed stream evaluation. "
                "No more data release available."
            )
        elif phase == "not_all_predicted":
            super().__init__(
                f"Algorithm {algo_id} has already predicted for this data segment, "
                "please wait for all other algorithms to predict"
            )
        else:
            super().__init__(f"Algorithm:{algo_id} current status is {status}.")
