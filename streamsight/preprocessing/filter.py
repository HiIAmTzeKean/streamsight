from abc import ABC, abstractmethod

import pandas as pd


class Filter(ABC):
    """Abstract baseclass for filter implementations

    A filter needs to implement an ``apply`` method,
    which takes as input a pandas DataFrame, and returns a processed pandas DataFrame.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Filter to the DataFrame passed.

        :param df: DataFrame to filter
        :type df: pd.DataFrame
        """
        raise NotImplementedError

    def __str__(self):
        attrs = self.__dict__
        return f"{self.__class__.__name__}({', '.join((f'{k}={v}' for k, v in attrs.items()))})"


class MinItemsPerUser(Filter):
    """Require that a user has interacted with a minimum number of items.
    
    This code is adapted from RecPack :cite:`recpack`
    
    Example
    ~~~~~~~~~
    
    .. code-block::
    
        Original interactions
        1 - a
        1 - b
        1 - c
        2 - a
        2 - b
        2 - d
        3 - a
        3 - b
        3 - d

        After MinItemsPerUser(3)
        1 - a
        1 - b
        2 - a
        2 - b
        3 - a
        3 - b


    :param min_items_per_user: Minimum number of items required.
    :type min_items_per_user: int
    :param item_ix: Name of the column in which item identifiers are listed.
    :type item_ix: str
    :param user_ix: Name of the column in which user identifiers are listed.
    :type user_ix: str
    :param count_duplicates: Count multiple interactions with the same item, defaults to True
    :type count_duplicates: bool
    """
    
    def __init__(
        self,
        min_items_per_user: int,
        item_ix: str,
        user_ix: str,
        count_duplicates: bool = True,
    ):
        self.min_iu = min_items_per_user
        self.count_duplicates = count_duplicates

        self.item_ix = item_ix
        self.user_ix = user_ix

    def apply(self, df) -> pd.DataFrame:
        uids = (
            df[self.user_ix]
            if self.count_duplicates
            else df.drop_duplicates([self.user_ix, self.item_ix])[self.user_ix]
        )
        cnt_items_per_user = uids.value_counts()
        users_of_interest = list(cnt_items_per_user[cnt_items_per_user >= self.min_iu].index)

        return df[df[self.user_ix].isin(users_of_interest)].copy()

    
class MinUsersPerItem(Filter):
    """Require that a minimum number of users has interacted with an item.
    
    This code is adapted from RecPack :cite:`recpack`
    
    Example
    ~~~~~~~~~
    
    .. code-block::
    
        Original interactions
        1 - a
        1 - b
        1 - c
        2 - a
        2 - b
        2 - d
        3 - a
        3 - b
        3 - d

        After MinUsersPerItem(3)
        1 - a
        1 - b
        1 - c
        2 - a
        2 - b
        2 - d
        3 - a
        3 - b
        3 - d

    :param min_users_per_item: Minimum number of users required.
    :type min_users_per_item: int
    :param item_ix: Name of the column in which item identifiers are listed.
    :type item_ix: str
    :param user_ix: Name of the column in which user identifiers are listed.
    :type user_ix: str
    :param count_duplicates: Count multiple interactions with the same user, defaults to True
    :type count_duplicates: bool
    """

    def __init__(
        self,
        min_users_per_item: int,
        item_ix: str,
        user_ix: str,
        count_duplicates: bool = True,
    ):

        self.item_ix = item_ix
        self.user_ix = user_ix

        self.min_ui = min_users_per_item
        self.count_duplicates = count_duplicates

    def apply(self, df) -> pd.DataFrame:
        iids = (
            df[self.item_ix]
            if self.count_duplicates
            else df.drop_duplicates([self.user_ix, self.item_ix])[self.item_ix]
        )
        cnt_users_per_item = iids.value_counts()
        items_of_interest = list(cnt_users_per_item[cnt_users_per_item >= self.min_ui].index)

        return df[df[self.item_ix].isin(items_of_interest)].copy()
