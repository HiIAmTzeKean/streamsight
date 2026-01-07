import logging
import time
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.itemknn import ItemKNN
from streamsight.matrix import InteractionMatrix

logger = logging.getLogger(__name__)


class ItemKNNIncremental(ItemKNN):
    """Incremental version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to allow for incremental updates
    to the model. The incremental updates are done by updating the historical
    data with the new data by appending the new data to the historical data.
    """

    def __init__(self, K=10):
        super().__init__(K)
        self.historical_data: InteractionMatrix

    def fit(self, X: InteractionMatrix) -> "Algorithm":
        start = time.time()
        if not hasattr(self, "historical_data"):
            self.historical_data = X.copy()
        else:
            logger.debug(f"Updating historical data for {self.name}")
            self.historical_data = self.historical_data + X
        end = time.time()
        logger.debug(
            f"Updated historical data for {self.name} - Took {end - start :.3}s"
        )
        super().fit(self.historical_data)
        return self
