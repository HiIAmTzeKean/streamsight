from dataclasses import dataclass

from streamsight.utils import get_data_dir


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    user_ix: str = "user_id"
    """Name of the column in the DataFrame with user identifiers"""
    item_ix: str = "item_id"
    """Name of the column in the DataFrame with item identifiers"""
    timestamp_ix: str = "timestamp"
    """Name of the column in the DataFrame that contains time of interaction in seconds since epoch."""
    dataset_url: str = "http://example.com"
    """URL to fetch the dataset from."""
    default_base_path: str = str(get_data_dir())
    """Default base path where the dataset will be stored."""
    remote_zipname: str = ""
    remote_filename: str = ""
    """Name of the file containing user interaction."""

    @property
    def default_filename(self) -> str:
        """Derived filename from remote components."""
        if not self.remote_zipname or not self.remote_filename:
            return "dataset.csv"
        return f"{self.remote_zipname}_{self.remote_filename}"


@dataclass
class MetadataConfig(DatasetConfig):
    sep: str = "|"
    """Column separator in the data file."""

    def __post_init__(self) -> None:
        self.default_base_path = super().default_base_path + "/metadata"

    @property
    def column_names(self) -> list[str]:
        """
        Ordered list of column names for pd.read_table.

        Returns:
            list[str]: Column names in file order [user_id, age, gender, ...]

        Example:
            ["userId", "age", "gender", "occupation", "zipcode"]
        """
        return []

    @property
    def dtype_dict(self) -> dict:
        """
        Data type mapping for all columns.

        Used in pd.read_table() dtype parameter to ensure correct
        column types are loaded from file.

        Returns:
            dict: Mapping of column names to numpy dtypes

        Example:
            {
                "age": "int64",
                "gender": "<U1",  # string
                "occupation": "object",
                "zipcode": "object"
            }
        """
        return {}
