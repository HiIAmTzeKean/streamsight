from abc import ABC, abstractmethod
from typing import Tuple

from deprecation import deprecated

from streamsight.matrix import InteractionMatrix


class Processor(ABC):
    """Base class for processing data.

    Abstract class for processing data. The `process` method should be
    implemented by the subclass to process the data. The programmer can
    create a subclass of preferred processing of data split.
    """

    def __init__(self):
        pass

    @abstractmethod
    def process(
        self,
        past_interaction: InteractionMatrix,
        future_interaction: InteractionMatrix,
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Injects the user ID to indicate ID for prediction.

        User ID to be predicted by the model will be indicated with item ID of
        "-1" as the corresponding label. The matrix with past interactions will
        contain the user ID to be predicted which will be derived from the set
        of user IDs in the future interaction matrix. Timestamp of the masked
        interactions will be preserved as the item ID are simply masked with
        "-1".

        :param past_interaction: Matrix of past interactions.
        :type past_interaction: InteractionMatrix
        :param future_interaction: Matrix of future interactions.
        :type future_interaction: InteractionMatrix
        :return: Tuple of past interaction with injected user ID to predict and
            ground truth future interactions of the actual interaction
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        pass


class PredictionDataProcessor(Processor):
    """Injects the user ID to indicate ID for prediction.

    Operates on the past and future interaction matrices to inject the user
    ID to be predicted by the model into the past interaction matrix. The
    resulting past interaction matrix will contain the user ID to be
    predicted which will be derived from the set of user IDs in the future
    interaction matrix. Timestamp of the masked interactions will be preserved as
    the item ID are simply masked with "-1".

    The corresponding ground truth future interactions of the actual interaction
    will be returned as well in a tuple when `process` is called.
    """

    def _inject_user_id(
        self,
        past_interaction: InteractionMatrix,
        future_interaction: InteractionMatrix,
        top_K: int = 1,
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Injects the user ID to indicate ID for prediction.

        User ID to be predicted by the model will be indicated with item ID of
        "-1" as the corresponding label. The matrix with past interactions will
        contain the user ID to be predicted which will be derived from the set
        of user IDs in the future interaction matrix. Timestamp of the masked
        interactions will be preserved as the item ID are simply masked with
        "-1".

        :param past_interaction: Matrix of past interactions.
        :type past_interaction: InteractionMatrix
        :param future_interaction: Matrix of future interactions.
        :type future_interaction: InteractionMatrix
        :return: Tuple of past interaction with injected user ID to predict and
            ground truth future interactions of the actual interaction
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        users_to_predict = future_interaction.get_users_n_first_interaction(
            top_K
        )
        masked_frame = users_to_predict.copy_df()
        masked_frame[InteractionMatrix.ITEM_IX] = InteractionMatrix.MASKED_LABEL
        return past_interaction.concat(masked_frame), users_to_predict

    @deprecated(
        deprecated_in="0.2.0",
        details="There is no need to inject item ID for with masked user ID.",
    )
    def _inject_item_id(
        self,
        past_interaction: InteractionMatrix,
        future_interaction: InteractionMatrix,
        top_K: int = 1,
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        """Injects the item ID to indicate ID for prediction.

        User ID to be predicted by the model will be indicated with item ID of
        "-1" as the corresponding label. The matrix with past interactions will
        contain the item ID to be predicted which will be derived from the set
        of item IDs in the future interaction matrix. Timestamp of the masked
        interactions will be preserved as the item ID are simply masked with
        "-1".

        :param past_interaction: Matrix of past interactions.
        :type past_interaction: InteractionMatrix
        :param future_interaction: Matrix of future interactions.
        :type future_interaction: InteractionMatrix
        :return: Tuple of past interaction with injected user ID to predict and
            ground truth future interactions of the actual interaction
        :rtype: Tuple[InteractionMatrix, InteractionMatrix]
        """
        items_to_predict = future_interaction.get_items_n_first_interaction(
            top_K
        )
        masked_frame = items_to_predict.copy_df()
        masked_frame[InteractionMatrix.USER_IX] = InteractionMatrix.MASKED_LABEL
        return past_interaction.concat(masked_frame), items_to_predict

    def process(
        self,
        past_interaction: InteractionMatrix,
        future_interaction: InteractionMatrix,
        top_K: int = 1,
    ) -> Tuple[InteractionMatrix, InteractionMatrix]:
        return self._inject_user_id(past_interaction,
                                    future_interaction,
                                    top_K)
