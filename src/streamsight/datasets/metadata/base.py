import logging
from typing import ClassVar, Optional

import pandas as pd

from streamsight.datasets.base import DataFetcher
from streamsight.datasets.config import MetadataConfig
from streamsight.utils.path import safe_dir


logger = logging.getLogger(__name__)


class Metadata(DataFetcher):
    config: ClassVar[MetadataConfig] = MetadataConfig()

    def __init__(
        self,
        filename: Optional[str] = None,
        base_path: Optional[str] = None,
    ) -> None:
        self.base_path = base_path if base_path else self.config.default_base_path
        logger.debug(f"{self.name} being initialized with '{self.base_path}' as the base path.")

        self.filename = filename if filename else self.config.default_filename
        if not self.filename:
            raise ValueError("No filename specified, and no default known.")

        safe_dir(self.base_path)
        logger.debug(f"{self.name} is initialized.")

    def load(self) -> pd.DataFrame:
        """Load the metadata from file and return it as a DataFrame.

        :return: Dataframe containing the metadata
        :rtype: pd.DataFrame
        """
        return self._load_dataframe()
