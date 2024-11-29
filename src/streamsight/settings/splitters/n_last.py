import logging

from streamsight.matrix import InteractionMatrix
from .base import Splitter


logger = logging.getLogger(__name__)


class NLastInteractionSplitter(Splitter):
    """Splits the n most recent interactions of a user into the second return value,
    and earlier interactions into the first.

    Args:
        n (int): Number of most recent actions to assign to the second return value.
        n_seq_data (int, optional): Number of last interactions to provide as unlabeled data
            for model to make prediction. Defaults to 1.

    Raises:
        ValueError: If n is less than 1, as this would cause the ground truth data to be empty.
    """

    def __init__(self, n: int, n_seq_data: int = 1) -> None:
        super().__init__()
        if n < 1:
            raise ValueError(
                f"n must be greater than 0, got {n}. "
                f"Values for n < 1 will cause the ground truth data to be empty."
            )
        self.n = n
        self.n_seq_data = n_seq_data

    def split(self, data: InteractionMatrix) -> tuple[InteractionMatrix, InteractionMatrix]:
        future_interaction = data.get_users_n_last_interaction(self.n)
        past_interaction = data - future_interaction
        past_interaction = past_interaction.get_users_n_last_interaction(self.n_seq_data)
        logger.debug(f"{self.identifier} has complete split")

        return past_interaction, future_interaction
