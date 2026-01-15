import logging
import operator
from collections.abc import Callable
from copy import deepcopy
from enum import StrEnum
from typing import Literal, overload
from warnings import warn

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .exception import TimestampAttributeMissingError


logger = logging.getLogger(__name__)


class ItemUserBasedEnum(StrEnum):
    """Enum class for item and user based properties.

    Enum class to indicate if the function or logic is based on item or user.
    """

    ITEM = "item"
    """Property based on item"""
    USER = "user"
    """Property based on user"""

    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check valid value for ItemUserBasedEnum

        :param value: String value input
        :type value: str
        """
        return value in ItemUserBasedEnum


class InteractionMatrix:
    """Matrix of interaction data between users and items.

    It provides a number of properties and methods for easy manipulation of this interaction data.

    .. attention::

        - The InteractionMatrix does not assume binary user-item pairs.
          If a user interacts with an item more than once, there will be two
          entries for this user-item pair.

        - We assume that the user and item IDs are integers starting from 0. IDs
          that are indicated by "-1" are reserved to label the user or item to
          be predicted. This assumption is crucial as it will be used during the
          split scheme and evaluation of the RS since it will affect the 2D shape
          of the CSR matrix

    :param df: Dataframe containing user-item interactions. Must contain at least
        item ids and user ids.
    :type df: pd.DataFrame
    :param item_ix: Item ids column name.
    :type item_ix: str
    :param user_ix: User ids column name.
    :type user_ix: str
    :param timestamp_ix: Interaction timestamps column name.
    :type timestamp_ix: str
    :param shape: The desired shape of the matrix, i.e. the number of users and items.
        If no shape is specified, the number of users will be equal to the
        maximum user id plus one, the number of items to the maximum item
        id plus one.
    :type shape: tuple[int, int], optional
    :param skip_df_processing: Skip processing of the dataframe. This is useful
        when the dataframe is already processed and the columns are already
        renamed.
    :type skip_df_processing: bool, optional
    """

    ITEM_IX = "iid"
    USER_IX = "uid"
    TIMESTAMP_IX = "ts"
    INTERACTION_IX = "interactionid"
    MASKED_LABEL = -1

    def __init__(
        self,
        df: pd.DataFrame,
        item_ix: str,
        user_ix: str,
        timestamp_ix: str,
        shape: None | tuple[int, int] = None,
        skip_df_processing: bool = False,
    ) -> None:
        self.user_item_shape: tuple[int, int]
        """The shape of the interaction matrix, i.e. `|user| x |item|`."""
        if shape:
            self.user_item_shape = shape

        if skip_df_processing:
            self._df = df
            return

        col_mapper = {
            item_ix: InteractionMatrix.ITEM_IX,
            user_ix: InteractionMatrix.USER_IX,
            timestamp_ix: InteractionMatrix.TIMESTAMP_IX,
        }
        df = df.rename(columns=col_mapper)
        required_columns = [
            InteractionMatrix.USER_IX,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.TIMESTAMP_IX,
        ]
        extra_columns = [col for col in df.columns if col not in required_columns]
        df = df[required_columns + extra_columns].copy()
        # TODO refactor this statement below
        df = df.reset_index(drop=True).reset_index().rename(columns={"index": InteractionMatrix.INTERACTION_IX})

        self._df = df

    def copy(self) -> "InteractionMatrix":
        """Create a deep copy of this InteractionMatrix.

        :return: Deep copy of this InteractionMatrix.
        :rtype: InteractionMatrix
        """
        return deepcopy(self)

    def copy_df(self, reset_index: bool = False) -> "pd.DataFrame":
        """Create a deep copy of the dataframe.

        :return: Deep copy of dataframe.
        :rtype: pd.DataFrame
        """
        if reset_index:
            return deepcopy(self._df.reset_index(drop=True))
        return deepcopy(self._df)

    def concat(self, im: "InteractionMatrix | pd.DataFrame") -> "InteractionMatrix":
        """Concatenate this InteractionMatrix with another.

        .. note::
            This is a inplace operation. and will modify the current object.

        :param im: InteractionMatrix to concat with.
        :type im: Union[InteractionMatrix, pd.DataFrame]
        :return: InteractionMatrix with the interactions from both matrices.
        :rtype: InteractionMatrix
        """
        if isinstance(im, pd.DataFrame):
            self._df = pd.concat([self._df, im])
        else:
            self._df = pd.concat([self._df, im._df])

        return self

    # TODO this should be shifted to prediction matrix
    def union(self, im: "InteractionMatrix") -> "InteractionMatrix":
        """Combine events from this InteractionMatrix with another.

        :param im: InteractionMatrix to union with.
        :type im: InteractionMatrix
        :return: Union of interactions in this InteractionMatrix and the other.
        :rtype: InteractionMatrix
        """
        return self + im

    def difference(self, im: "InteractionMatrix") -> "InteractionMatrix":
        """Difference between this InteractionMatrix and another.

        :param im: InteractionMatrix to subtract from this.
        :type im: InteractionMatrix
        :return: Difference between this InteractionMatrix and the other.
        :rtype: InteractionMatrix
        """
        return self - im

    @property
    def values(self) -> csr_matrix:
        """All user-item interactions as a sparse matrix of size (|`global_users`|, |`global_items`|).

        Each entry is the number of interactions between that user and item.
        If there are no interactions between a user and item, the entry is 0.

        :return: Interactions between users and items as a csr_matrix.
        :rtype: csr_matrix
        """
        # TODO issue with -1 labeling in the interaction matrix should i create prediction matrix
        if not hasattr(self, "user_item_shape"):
            raise AttributeError("InteractionMatrix has no `user_item_shape` attribute. Please call mask_shape() first.")

        values = np.ones(self._df.shape[0])
        indices = self._df[[InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]].values
        indices = (indices[:, 0], indices[:, 1])

        matrix = csr_matrix((values, indices), shape=self.user_item_shape, dtype=np.int32)
        return matrix

    @property
    def indices(self) -> tuple[list[int], list[int]]:
        """Returns a tuple of lists of user IDs and item IDs corresponding to interactions.

        :return: tuple of lists of user IDs and item IDs that correspond to at least one interaction.
        :rtype: tuple[list[int], list[int]]
        """
        return self.values.nonzero()

    def nonzero(self) -> tuple[list[int], list[int]]:
        return self.values.nonzero()

    @overload
    def users_in(self, U: set[int]) -> "InteractionMatrix": ...
    @overload
    def users_in(self, U: set[int], inplace: Literal[False]) -> "InteractionMatrix": ...
    @overload
    def users_in(self, U: set[int], inplace: Literal[True]) -> None: ...
    def users_in(self, U: set[int], inplace: bool = False) -> "None | InteractionMatrix":
        """Keep only interactions by one of the specified users.

        :param U: A set or list of users to select the interactions from.
        :type U: Union[set[int]
        :param inplace: Apply the selection in place or not, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        logger.debug("Performing users_in comparison")

        mask = self._df[InteractionMatrix.USER_IX].isin(U)

        return self._apply_mask(mask, inplace=inplace)

    @overload
    def _apply_mask(self, mask: pd.Series) -> "InteractionMatrix": ...
    @overload
    def _apply_mask(self, mask: pd.Series, inplace: Literal[True]) -> None: ...
    @overload
    def _apply_mask(self, mask: pd.Series, inplace: Literal[False]) -> "InteractionMatrix": ...
    def _apply_mask(self, mask: pd.Series, inplace: bool = False) -> "None | InteractionMatrix":
        interaction_m = self if inplace else self.copy()
        interaction_m._df = interaction_m._df[mask]
        return None if inplace else interaction_m

    def _timestamps_cmp(self, op: Callable, timestamp: float, inplace: bool = False) -> "None | InteractionMatrix":
        """Filter interactions based on timestamp.
        Keep only interactions for which op(t, timestamp) is True.

        :param op: Comparison operator.
        :type op: Callable
        :param timestamp: Timestamp to compare against in seconds from epoch.
        :type timestamp: float
        :param inplace: Modify the data matrix in place. If False, returns a new object.
        :type inplace: bool, optional
        """
        logger.debug(f"Performing {op.__name__}(t, {timestamp})")

        mask = op(self._df[InteractionMatrix.TIMESTAMP_IX], timestamp)
        return self._apply_mask(mask, inplace=inplace)

    @overload
    def timestamps_gt(self, timestamp: float) -> "InteractionMatrix": ...
    @overload
    def timestamps_gt(self, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_gt(self, timestamp: float, inplace: bool = False) -> "None | InteractionMatrix":
        """Select interactions after a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.gt, timestamp, inplace)

    @overload
    def timestamps_gte(self, timestamp: float) -> "InteractionMatrix": ...
    @overload
    def timestamps_gte(self, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_gte(self, timestamp: float, inplace: bool = False) -> "None | InteractionMatrix":
        """Select interactions after and including a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.ge, timestamp, inplace)

    @overload
    def timestamps_lt(self, timestamp: float) -> "InteractionMatrix": ...
    @overload
    def timestamps_lt(self, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_lt(self, timestamp: float, inplace: bool = False) -> "None | InteractionMatrix":
        """Select interactions up to a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.lt, timestamp, inplace)

    @overload
    def timestamps_lte(self, timestamp: float) -> "InteractionMatrix": ...
    @overload
    def timestamps_lte(self, timestamp: float, inplace: Literal[True]) -> None: ...
    def timestamps_lte(self, timestamp: float, inplace: bool = False) -> "None | InteractionMatrix":
        """Select interactions up to and including a given timestamp.

        :param timestamp: The timestamp with which
            the interactions timestamp is compared.
        :type timestamp: float
        :param inplace: Apply the selection in place if True, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        return self._timestamps_cmp(operator.le, timestamp, inplace)

    def __add__(self, im: "InteractionMatrix") -> "InteractionMatrix":
        """Combine events from this InteractionMatrix with another.

        :param im: InteractionMatrix to union with.
        :type im: InteractionMatrix
        :return: Union of interactions in this InteractionMatrix and the other.
        :rtype: InteractionMatrix
        """
        df = pd.concat([self._df, im._df], copy=False)

        shape = None
        if hasattr(self, "user_item_shape") and hasattr(im, "user_item_shape"):
            shape = (max(self.user_item_shape[0], im.user_item_shape[0]), max(self.user_item_shape[1], im.user_item_shape[1]))
            self.user_item_shape = shape

        return InteractionMatrix(
            df,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
            InteractionMatrix.TIMESTAMP_IX,
            shape,
            True,
        )

    def __sub__(self, im: "InteractionMatrix") -> "InteractionMatrix":
        full_data = pd.MultiIndex.from_frame(self._df)
        data_part_2 = pd.MultiIndex.from_frame(im._df)
        data_part_1 = full_data.difference(data_part_2).to_frame().reset_index(drop=True)

        shape = None
        if hasattr(self, "user_item_shape") and hasattr(im, "user_item_shape"):
            shape = (max(self.user_item_shape[0], im.user_item_shape[0]), max(self.user_item_shape[1], im.user_item_shape[1]))

        return InteractionMatrix(
            data_part_1,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
            InteractionMatrix.TIMESTAMP_IX,
            shape,
            True,
        )

    def __repr__(self) -> str:
        return repr(self._df)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, InteractionMatrix):
            logger.debug(f"Comparing {type(value)} with InteractionMatrix is not supported")
            return False
        return self._df.equals(value._df)

    def __len__(self) -> int:
        """Return the number of interactions in the matrix.

        This is distinct from the shape of the matrix, which is the number of
        users and items that has been released to the model. The length of the
        matrix is the number of interactions present in the matrix resulting
        from filter operations.
        """
        return len(self._df)

    @overload
    def items_in(self, id_set: set[int]) -> "InteractionMatrix": ...
    @overload
    def items_in(self, id_set: set[int], inplace: Literal[False]) -> "InteractionMatrix": ...
    @overload
    def items_in(self, id_set: set[int], inplace: Literal[True]) -> None: ...
    def items_in(self, id_set: set[int], inplace=False) -> "None | InteractionMatrix":
        """Keep only interactions with the specified items.

        :param id_set: A set or list of items to select the interactions.
        :type id_set: set[int]
        :param inplace: Apply the selection in place or not, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        logger.debug("Performing items_in comparison")

        mask = self._df[InteractionMatrix.ITEM_IX].isin(id_set)

        return self._apply_mask(mask, inplace=inplace)

    @overload
    def items_not_in(self, id_set: set[int]) -> "InteractionMatrix": ...
    @overload
    def items_not_in(self, id_set: set[int], inplace: Literal[False]) -> "InteractionMatrix": ...
    @overload
    def items_not_in(self, id_set: set[int], inplace: Literal[True]) -> None: ...
    def items_not_in(self, id_set: set[int], inplace: bool=False) -> "None | InteractionMatrix":
        """Keep only interactions not with the specified items.

        :param id_set: A set or list of items to exclude from the interactions.
        :type id_set: set[int]
        :param inplace: Apply the selection in place or not, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        logger.debug("Performing items_not_in comparison")

        mask = ~self._df[InteractionMatrix.ITEM_IX].isin(id_set)

        return self._apply_mask(mask, inplace=inplace)

    def users_not_in(self, U: set[int], inplace=False) -> "None | InteractionMatrix":
        """Keep only interactions not by the specified users.

        :param U: A set or list of users to exclude from the interactions.
        :type U: set[int]
        :param inplace: Apply the selection in place or not, defaults to False
        :type inplace: bool, optional
        :return: None if `inplace`, otherwise returns a new InteractionMatrix object
        :rtype: Union[InteractionMatrix, None]
        """
        logger.debug("Performing users_not_in comparison")

        mask = ~self._df[InteractionMatrix.USER_IX].isin(U)

        return self._apply_mask(mask, inplace=inplace)

    def interactions_in(self, interaction_ids: list[int], inplace: bool = False) -> "None | InteractionMatrix":
        """Select the interactions by their interaction ids

        :param interaction_ids: A list of interaction ids
        :type interaction_ids: list[int]
        :param inplace: Apply the selection in place,
            or return a new InteractionMatrix object, defaults to False
        :type inplace: bool, optional
        :return: None if inplace, otherwise new InteractionMatrix
            object with the selected interactions
        :rtype: Union[None, InteractionMatrix]
        """
        logger.debug("Performing interactions_in comparison")

        mask = self._df[InteractionMatrix.INTERACTION_IX].isin(interaction_ids)

        unknown_interaction_ids = set(interaction_ids).difference(self._df[InteractionMatrix.INTERACTION_IX].unique())

        if unknown_interaction_ids:
            warn(f"IDs {unknown_interaction_ids} not present in data")
        if not interaction_ids:
            warn("No interaction IDs given, returning empty InteractionMatrix.")

        return self._apply_mask(mask, inplace=inplace)

    def _get_last_n_interactions(
        self,
        by: ItemUserBasedEnum,
        n_seq_data: int,
        t_upper: None | int = None,
        id_in: None | set[int] = None,
        inplace=False,
    ) -> "InteractionMatrix":
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        if t_upper is None:
            t_upper = self.max_timestamp + 1  # to include the last timestamp

        interaction_m = self if inplace else self.copy()

        mask = interaction_m._df[InteractionMatrix.TIMESTAMP_IX] < t_upper
        if id_in and by == ItemUserBasedEnum.USER:
            mask = mask & interaction_m._df[InteractionMatrix.USER_IX].isin(id_in)
        elif id_in and by == ItemUserBasedEnum.ITEM:
            mask = mask & interaction_m._df[InteractionMatrix.ITEM_IX].isin(id_in)

        if by == ItemUserBasedEnum.USER:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.USER_IX).tail(n_seq_data)
        else:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.ITEM_IX).tail(n_seq_data)
        interaction_m._df = c_df

        return interaction_m

    def _get_first_n_interactions(
        self, by: ItemUserBasedEnum, n_seq_data: int, t_lower: None | int = None, inplace=False
    ) -> "InteractionMatrix":
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        if t_lower is None:
            t_lower = self.min_timestamp

        interaction_m = self if inplace else self.copy()

        mask = interaction_m._df[InteractionMatrix.TIMESTAMP_IX] >= t_lower
        if by == ItemUserBasedEnum.USER:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.USER_IX).head(n_seq_data)
        else:
            c_df = interaction_m._df[mask].groupby(InteractionMatrix.ITEM_IX).head(n_seq_data)
        interaction_m._df = c_df
        return interaction_m

    def get_users_n_last_interaction(
        self,
        n_seq_data: int = 1,
        t_upper: None | int = None,
        user_in: None | set[int] = None,
        inplace: bool = False,
    ) -> "InteractionMatrix":
        """Select the last n interactions for each user.

        :param n_seq_data: Number of interactions to select, defaults to 1
        :type n_seq_data: int, optional
        :param t_upper: Seconds past t. Upper limit for the timestamp
            of the interactions to select, defaults to None
        :type t_upper: None | int, optional
        :param user_in: set of user IDs to select the interactions from,
            defaults to None
        :type user_in: None | set[int], optional
        :param inplace: If operation is inplace, defaults to False
        :type inplace: bool, optional
        :return: Resulting interaction matrix
        :rtype: InteractionMatrix
        """
        logger.debug("Performing get_user_n_last_interaction comparison")
        return self._get_last_n_interactions(ItemUserBasedEnum.USER, n_seq_data, t_upper, user_in, inplace)

    def get_items_n_last_interaction(
        self,
        n_seq_data: int = 1,
        t_upper: None | int = None,
        item_in: None | set[int] = None,
        inplace: bool = False,
    ) -> "InteractionMatrix":
        """Select the last n interactions for each item.

        :param n_seq_data: Number of interactions to select, defaults to 1
        :type n_seq_data: int, optional
        :param t_upper: Seconds past t. Upper limit for the timestamp
            of the interactions to select, defaults to None
        :type t_upper: None | int, optional
        :param item_in: set of item IDs to select the interactions from,
            defaults to None
        :type item_in: None | set[int], optional
        :param inplace: If operation is inplace, defaults to False
        :type inplace: bool, optional
        :return: Resulting interaction matrix
        :rtype: InteractionMatrix
        """
        logger.debug("Performing get_item_n_last_interaction comparison")
        return self._get_last_n_interactions(ItemUserBasedEnum.ITEM, n_seq_data, t_upper, item_in, inplace)

    def get_users_n_first_interaction(
        self, n_seq_data: int = 1, t_lower: None | int = None, inplace=False
    ) -> "InteractionMatrix":
        """Select the first n interactions for each user.

        :param n_seq_data: Number of interactions to select, defaults to 1
        :type n_seq_data: int, optional
        :param t_lower: Seconds past t. Lower limit for the timestamp
            of the interactions to select, defaults to None
        :type t_lower: None | int, optional
        :param inplace: If operation is inplace, defaults to False
        :type inplace: bool, optional
        :return: Resulting interaction matrix
        :rtype: InteractionMatrix
        """
        return self._get_first_n_interactions(ItemUserBasedEnum.USER, n_seq_data, t_lower, inplace)

    def get_items_n_first_interaction(
        self, n_seq_data: int = 1, t_lower: None | int = None, inplace=False
    ) -> "InteractionMatrix":
        """Select the first n interactions for each item.

        :param n_seq_data: Number of interactions to select, defaults to 1
        :type n_seq_data: int, optional
        :param t_lower: Seconds past t. Lower limit for the timestamp
            of the interactions to select, defaults to None
        :type t_lower: None | int, optional
        :param inplace: If operation is inplace, defaults to False
        :type inplace: bool, optional
        :return: Resulting interaction matrix
        :rtype: InteractionMatrix
        """
        return self._get_first_n_interactions(ItemUserBasedEnum.ITEM, n_seq_data, t_lower, inplace)

    @property
    def item_interaction_sequence_matrix(self) -> csr_matrix:
        """Converts the interaction data into an item interaction sequence matrix.

        Dataframe values are converted into such that the row sequence is maintained and
        the item that interacted with will have the column at item_id marked with 1.
        """
        values = np.ones(self._df.shape[0])
        indices = (np.arange(self._df.shape[0]), self._df[InteractionMatrix.ITEM_IX].to_numpy())
        shape = (self._df.shape[0], self.user_item_shape[1])

        sparse_matrix = csr_matrix((values, indices), shape=shape, dtype=values.dtype)
        return sparse_matrix

    @property
    def user_id_sequence_array(self) -> np.ndarray:
        """Array of user IDs in the order of interactions.

        :return: Numpy array of user IDs.
        :rtype: np.ndarray
        """
        return self._df[InteractionMatrix.USER_IX].to_numpy()

    def get_prediction_data(self) -> "InteractionMatrix":
        """Get the data to be predicted.

        :return: InteractionMatrix with only the data to be predicted.
        :rtype: InteractionMatrix
        """
        return self.items_in({-1})

    def get_interaction_data(self) -> "InteractionMatrix":
        """Get the data that is not denoted by "-1".

        User and item IDs that are not denoted by "-1" are the ones that are
        known to the model.

        :return: InteractionMatrix with only the known data.
        :rtype: InteractionMatrix
        """
        mask = (self._df[InteractionMatrix.USER_IX] != -1) & (self._df[InteractionMatrix.ITEM_IX] != -1)
        return self._apply_mask(mask)

    @property
    def user_ids(self) -> set[int]:
        """The set of all user ID in matrix"""
        return set(self._df[InteractionMatrix.USER_IX].dropna().unique())

    @property
    def item_ids(self) -> set[int]:
        """The set of all item ID in matrix"""
        return set(self._df[InteractionMatrix.ITEM_IX].dropna().unique())

    @property
    def num_interactions(self) -> int:
        """The total number of interactions.

        :return: Total interaction count.
        :rtype: int
        """
        return len(self._df)

    @property
    def has_timestamps(self) -> bool:
        """Boolean indicating whether instance has timestamp information.

        :return: True if timestamps information is available, False otherwise.
        :rtype: bool
        """
        return self.TIMESTAMP_IX in self._df

    @property
    def min_timestamp(self) -> int:
        """The earliest timestamp in the interaction

        :return: The earliest timestamp.
        :rtype: int
        """
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        return self._df[self.TIMESTAMP_IX].min()

    @property
    def max_timestamp(self) -> int:
        """The latest timestamp in the interaction

        :return: The latest timestamp.
        :rtype: int
        """
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        return self._df[self.TIMESTAMP_IX].max()

    @property
    def max_global_user_id(self) -> int:
        """The highest known global user ID in the interaction matrix.

        The difference between `max_global_user_id` and `max_user_id` is that
        `max_global_user_id` considers all user IDs present in the dataframe,
        including users that are only encountered during prediction time.
        """
        return max(int(self._df[InteractionMatrix.USER_IX].max()) + 1, self.user_item_shape[0])

    @property
    def max_global_item_id(self) -> int:
        return max(int(self._df[InteractionMatrix.ITEM_IX].max()) + 1, self.user_item_shape[1])

    @property
    def max_known_user_id(self) -> int:
        """The highest known user ID in the interaction matrix."""
        max_val = self._df[(self._df != -1).all(axis=1)][InteractionMatrix.USER_IX].max()
        if pd.isna(max_val):
            return self.user_item_shape[0]
        return min(int(max_val) + 1, self.user_item_shape[0])

    @property
    def max_known_item_id(self) -> int:
        """The highest known user ID in the interaction matrix."""
        max_val = self._df[(self._df != -1).all(axis=1)][InteractionMatrix.ITEM_IX].max()
        if pd.isna(max_val):
            return self.user_item_shape[1]
        return min(int(max_val) + 1, self.user_item_shape[1])

    @property
    def max_user_id(self) -> int:
        """The highest known user ID in the interaction matrix.

        :return: The highest user ID.
        :rtype: int
        """
        max_val = self._df[self._df != -1][InteractionMatrix.USER_IX].max()
        if np.isnan(max_val):
            return -1
        return max_val

    @property
    def max_item_id(self) -> int:
        """The highest known item ID in the interaction matrix.

        In the case of an empty matrix, the highest item ID is -1. This is
        consistent with the the definition that -1 denotes the item that is
        unknown. It would be incorrect to use any other value, since 0 is a
        valid item ID.

        :return: The highest item ID.
        :rtype: int
        """
        max_val = self._df[self._df != -1][InteractionMatrix.ITEM_IX].max()
        if np.isnan(max_val):
            return -1
        return max_val

    @property
    def timestamps(self) -> pd.Series:
        """The interaction timestamps indexed by user and item ID.

        :raises
        """
        """Timestamps of interactions as a pandas Series, indexed by user ID and item ID.

        :raises TimestampAttributeMissingError: If timestamp column is missing.
        :return: Interactions with composite index on (user ID, item ID)
        :rtype: pd.Series
        """
        if not self.has_timestamps:
            raise TimestampAttributeMissingError()
        index = pd.MultiIndex.from_frame(self._df[[InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX]])
        return pd.DataFrame(self._df[[InteractionMatrix.TIMESTAMP_IX]]).set_index(index)[InteractionMatrix.TIMESTAMP_IX]

    @property
    def latest_interaction_timestamps_matrix(self) -> csr_matrix:
        """A csr matrix containing the last interaction timestamp for each user, item pair.

        We only account for the last interacted timestamp making the dataset non-deduplicated.
        """
        timestamps = self.timestamps.groupby(["uid", "iid"]).max().reset_index()
        timestamp_mat = csr_matrix(
            (timestamps.ts.values, (timestamps.uid.values, timestamps.iid.values)),
            shape=self.user_item_shape,
        )

        return timestamp_mat
