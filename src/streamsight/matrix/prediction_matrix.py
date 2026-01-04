import logging
from warnings import warn

import numpy as np
import pandas as pd

from .interaction_matrix import InteractionMatrix


logger = logging.getLogger(__name__)


class PredictionMatrix(InteractionMatrix):
    @classmethod
    def from_interaction_matrix(cls, im: InteractionMatrix) -> "PredictionMatrix":
        """Create a PredictionMatrix from an InteractionMatrix.

        :param im: The InteractionMatrix to convert.
        :type im: InteractionMatrix
        :return: A new PredictionMatrix with the same data.
        :rtype: PredictionMatrix
        """
        return cls(
            df=im._df,
            item_ix=im.ITEM_IX,
            user_ix=im.USER_IX,
            timestamp_ix=im.TIMESTAMP_IX,
            shape=getattr(im, 'shape', None),
            skip_df_processing=True,
        )

    def mask_user_item_shape(
        self,
        shape: None | tuple[int, int] = None,
        drop_unknown_user: bool = False,
        drop_unknown_item: bool = False,
        inherit_max_id: bool = False,
    ) -> None:
        """Masks global user and item ID.

        To ensure released matrix released to the models only contains data
        that is intended to be released. This addresses the data leakage issue.
        It is recommended that the programmer defines the shape of the matrix
        such that the model only sees the data that is intended to be seen.

        =======
        Example
        =======

        Given the following case where the data is as follows::

            > uid: [0, 1, 2, 3, 4, 5]
            > iid: [0, 1, 2, 3, -1, -1]
            > ts : [0, 1, 2, 3, 4, 6]

        Where user 4, 5 is the user to be predicted. Assuming that user 4, 5 is an
        unknown user, that is, the model has never seen user 4, 5 before. The shape
        of the matrix should be (4, 4). This should be defined when calling the
        function in :param:`shape`.

        If the shape is defined, and it contains ID of unknown user/item, a warning
        will be raised if :attr:`drop_unknown` is set to False. If :attr:`drop_unknown`
        is set to True, the unknown user/item will be dropped from the data. All
        user/item ID greater than `shape[0]` will be dropped. This follows from
        the initial assumption that the user/item ID starts from 0 as defined in
        the dataset class.

        Else, in the event that :param:`shape` is not defined, the shape will be
        inferred from the data. The shape will be determined by the number of
        unique users/items. In this case the shape will be (5, 4). Note that the
        shape may not be as intended by the programmer if the data contains
        unknown users/items or if the dataframe does not contain all historical
        users/items.

        :param shape: Shape of the known user and item base. This value is
            usually set by the evaluator during the evaluation run. This value
            can also be set manually but the programmer if there is a need to
            alter the known user/item base. Defaults to None
        :type shape: Optional[tuple[int, int]], optional
        :param drop_unknown_user: To drop unknown users in the dataset,
            defaults to False
        :type drop_unknown_user: bool, optional
        :param drop_unknown_item: To drop unknown items in the dataset,
            defaults to False
        :type drop_unknown_item: bool, optional
        :param inherit_max_id: To inherit the maximum user and item ID from the
            given shape and the dataframe. This is useful when the shape is
            defined and the dataframe contains unknown users/items. Defaults to False
        :type inherit_max_id: bool, optional
        """

        if not shape:
            # infer shape from the data
            known_user = np.nan_to_num(self._df[self._df != -1][InteractionMatrix.USER_IX].max(), nan=-1)
            known_item = np.nan_to_num(self._df[self._df != -1][InteractionMatrix.ITEM_IX].max(), nan=-1)
            self.user_item_shape = (known_user, known_item)
            logger.debug(f"(user x item) shape inferred is {self.user_item_shape}")
            if known_user == -1 or known_item == -1:
                warn(
                    "One of the dimensions of the shape cannot be inferred from the data. "
                    "Call mask_shape() with shape parameter.",
                    stacklevel=2,
                )
            return

        logger.debug(
            f"(user x item) shape defined is {shape}. "
            f"Shape of dataframe stored in matrix was {self._df.shape} before masking"
        )
        if drop_unknown_user:
            logger.debug("Dropping unknown users from interaction matrix based on defined shape")
            self._df = pd.DataFrame(self._df[self._df[InteractionMatrix.USER_IX] < shape[0]])
        if drop_unknown_item:
            logger.debug("Dropping unknown items from interaction matrix based on defined shape")
            self._df = pd.DataFrame(self._df[self._df[InteractionMatrix.ITEM_IX] < shape[1]])
        logger.debug(f"Shape of dataframe stored in matrix is now {self._df.shape} after masking")

        if inherit_max_id:
            # we are only concerned about the absolute maximum id in the data regardless if its unknown
            known_user = int(self._df[InteractionMatrix.USER_IX].max())
            known_item = int(self._df[InteractionMatrix.ITEM_IX].max())
            # + 1 as id starts from 0
            self.user_item_shape = (max(shape[0], known_user + 1), max(shape[1], known_item + 1))
        else:
            self.user_item_shape = shape
        logger.debug(f"Final (user x item) shape defined is {self.user_item_shape}")
        self._check_user_item_shape()

    def _check_user_item_shape(self) -> None:
        if not hasattr(self, "user_item_shape"):
            raise AttributeError("InteractionMatrix has no `user_item_shape` attribute. Please call mask_shape() first.")
        if self.user_item_shape[0] is None or self.user_item_shape[1] is None:
            raise ValueError("Shape must be defined.")

        valid_df = self._df[self._df != -1]
        req_rows = valid_df[InteractionMatrix.USER_IX].max()
        req_cols = np.nan_to_num(valid_df[InteractionMatrix.ITEM_IX].max(), nan=-1)

        if self.user_item_shape[0] < req_rows or self.user_item_shape[1] < req_cols:
            logger.warning(
                "InteractionMatrix shape mismatch detected. "
                "Current shape: %s. Required minimum: (%s, %s). "
                "Data loss may occur.",
                self.user_item_shape,
                req_rows,
                req_cols,
            )
            warn(
                "Provided shape does not match known id; there are id that are out of bounds. "
                "Call mask_shape(drop_unknown=True) to drop unknown users and items.",
                category=UserWarning,
                stacklevel=2,
            )
