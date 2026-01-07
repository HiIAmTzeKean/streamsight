import logging
from typing import Optional


from streamsight.matrix import InteractionMatrix
from streamsight.settings import Setting
from streamsight.settings.splitters import (
    NLastInteractionSplitter,
)

logger = logging.getLogger(__name__)


class LeaveNOutSetting(Setting):
    def __init__(
        self,
        n_seq_data: int = 1,
        N: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.n_seq_data = n_seq_data
        # we use top_K to denote the number of items to predict
        self.top_K = N

        logger.info(f"Splitting data ")

        self._splitter = NLastInteractionSplitter(N, n_seq_data)

    def _split(self, data: InteractionMatrix):
        """Splits your dataset into a training, validation and test dataset
            based on the timestamp of the interaction.

        :param data: Interaction matrix to be split. Must contain timestamps.
        :type data: InteractionMatrix
        """

        self._background_data, future_interaction = self._splitter.split(data)
        # we need to copy the data to avoid modifying the background data
        past_interaction = self._background_data.copy()
        
        self._unlabeled_data, self._ground_truth_data = (
            self.prediction_data_processor.process(
                past_interaction, future_interaction, self.top_K
            )
        )
        self._t_window = None

    @property
    def params(self):
        """Parameters of the setting."""
        return {
            "n_seq_data": self.n_seq_data,
            "N": self.top_K,
        }
