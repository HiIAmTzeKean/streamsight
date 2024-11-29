from dataclasses import dataclass

from .base import DatasetConfig


@dataclass
class YelpDatasetConfig(DatasetConfig):
    """Yelp dataset configuration.

    Note: Yelp dataset must be manually downloaded from https://www.yelp.com/dataset/download
    as it requires acceptance of a license agreement. The dataset should be converted
    from JSON to CSV and placed in the data directory.
    """

    user_ix: str = "user_id"
    item_ix: str = "business_id"
    timestamp_ix: str = "date"
    rating_ix: str = "stars"
    dataset_url: str = "https://www.yelp.com/dataset/download"
    remote_filename: str = "yelp_academic_dataset_review.csv"

    @property
    def default_filename(self) -> str:
        """Return the default filename for Yelp dataset."""
        return self.remote_filename
