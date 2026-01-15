import logging
from dataclasses import dataclass, field
from enum import StrEnum

from streamsight.matrix import InteractionMatrix


logger = logging.getLogger(__name__)


class MetricLevelEnum(StrEnum):
    MICRO = "micro"
    MACRO = "macro"
    WINDOW = "window"
    USER = "user"

    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check valid value for MetricLevelEnum.

        Args:
            value: String value input.

        Returns:
            Whether the value is valid.
        """
        return value in MetricLevelEnum


@dataclass
class UserItemBaseStatus:
    """Unknown and known user/item base.

    This class is used to store the status of the user and item base. The class
    stores the known and unknown user and item set. The class also provides
    methods to update the known and unknown user and item set.
    """

    unknown_user: set[int] = field(default_factory=set)
    known_user: set[int] = field(default_factory=set)
    unknown_item: set[int] = field(default_factory=set)
    known_item: set[int] = field(default_factory=set)

    @property
    def known_shape(self) -> tuple[int, int]:
        """Known shape of the user-item interaction matrix.

        This is the shape of the released user/item interaction matrix to the
        algorithm. This shape follows from assumption in the dataset that
        ID increment in the order of time.

        Returns:
            Tuple of (|user|, |item|).
        """
        return (len(self.known_user), len(self.known_item))

    @property
    def global_shape(self) -> tuple[int, int]:
        """Global shape of the user-item interaction matrix.

        This is the shape of the user-item interaction matrix considering all
        the users and items that has been possibly exposed. The global shape
        considers the fact that an unknown user/item can be exposed during the
        prediction stage when an unknown user/item id is requested for prediction
        on the algorithm.

        Returns:
            Tuple of (|user|, |item|).
        """
        return (
            len(self.known_user) + len(self.unknown_user),
            len(self.known_item) + len(self.unknown_item),
        )

    @property
    def global_user_ids(self) -> set[int]:
        """Set of global user ids.

        Returns the set of global user ids. The global user ids are the union of
        known and unknown user ids.

        Returns:
            set[int]: Set of global user ids.
        """
        return self.known_user.union(self.unknown_user)

    @property
    def global_item_ids(self) -> set[int]:
        """Set of global item ids.

        Returns the set of global item ids. The global item ids are the union of
        known and unknown item ids.

        Returns:
            set[int]: Set of global item ids.
        """
        return self.known_item.union(self.unknown_item)

    def update_known_user_item_base(self, data: InteractionMatrix) -> None:
        """Updates the known user and item set with the data.

        Args:
            data (InteractionMatrix): Data to update the known user and item set with.
        """
        self.known_item.update(data.item_ids)
        self.known_user.update(data.user_ids)

    def update_unknown_user_item_base(self, data: InteractionMatrix) -> None:
        """Updates the unknown user and item set with the data.

        Args:
            data (InteractionMatrix): Data to update the unknown user and item set with.
        """
        self.unknown_user = data.user_ids.difference(self.known_user)
        self.unknown_item = data.item_ids.difference(self.known_item)

    def reset_unknown_user_item_base(self) -> None:
        """Clears the unknown user and item set.

        This method clears the unknown user and item set. This method should be
        called after the Phase 3 when the data release is done.
        """
        self.unknown_user = set()
        self.unknown_item = set()
