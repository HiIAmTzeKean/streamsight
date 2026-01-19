import logging

from .itemknn import ItemKNN


logger = logging.getLogger(__name__)


class ItemKNNRolling(ItemKNN):
    """Rolling version of ItemKNN algorithm.

    This class extends the ItemKNN algorithm to update the memory of the model
    to only keep the last window of interactions. The model is simply discarding
    all interactions that are older than the window size.
    """

    IS_BASE: bool = False
