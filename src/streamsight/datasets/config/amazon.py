from dataclasses import dataclass

from .base import DatasetConfig, MetadataConfig


@dataclass
class AmazonDatasetConfig(DatasetConfig):
    """Amazon dataset base configuration."""

    user_ix: str = "user_id"
    item_ix: str = "parent_asin"
    timestamp_ix: str = "timestamp"
    rating_ix: str = "rating"
    helpful_vote_ix: str = "helpful_vote"
    dataset_url: str = ""  # Set per subclass
    remote_filename: str = ""  # Set per subclass
    default_base_path: str = DatasetConfig.default_base_path + "/amazon"

    @property
    def default_filename(self) -> str:
        """Return just the filename for Amazon datasets (no zipname prefix)."""
        return self.remote_filename


@dataclass
class AmazonMusicDatasetConfig(AmazonDatasetConfig):
    """Amazon Music dataset configuration."""

    remote_filename: str = "Digital_Music.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Digital_Music.jsonl.gz"


@dataclass
class AmazonMovieDatasetConfig(AmazonDatasetConfig):
    """Amazon Movie dataset configuration."""

    remote_filename: str = "Movies_and_TV.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz"


@dataclass
class AmazonSubscriptionBoxesDatasetConfig(AmazonDatasetConfig):
    """Amazon Subscription Boxes dataset configuration."""

    remote_filename: str = "Subscription_Boxes.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Subscription_Boxes.jsonl.gz"


@dataclass
class AmazonBookDatasetConfig(AmazonDatasetConfig):
    """Amazon Books dataset configuration."""

    remote_filename: str = "Books.jsonl.gz"
    dataset_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz"


@dataclass
class AmazonItemMetadataConfig(MetadataConfig, AmazonDatasetConfig):
    """
    Amazon Item Metadata Base Configuration.

    Handles configuration for Amazon product metadata including:
    - Product identifiers (ASIN)
    - Product information (title, category, price, rating)
    - Rich content (features, description, images, videos)
    - Relational data (store, details, bought together)

    All properties are computed from base fields to ensure consistency.
    """

    item_ix: str = "parent_asin"
    """Name of the column containing product identifiers (parent ASIN)."""
    main_category_ix: str = "main_category"
    """Name of the column containing the main product category."""
    title_ix: str = "title"
    """Name of the column containing product title."""
    average_rating_ix: str = "average_rating"
    """Name of the column containing average product rating (0-5)."""
    rating_number_ix: str = "rating_number"
    """Name of the column containing number of ratings received."""
    features_ix: str = "features"
    """Name of the column containing product features (list)."""
    description_ix: str = "description"
    """Name of the column containing product description (list)."""
    price_ix: str = "price"
    """Name of the column containing product price."""
    images_ix: str = "images"
    """Name of the column containing product images URLs (list)."""
    videos_ix: str = "videos"
    """Name of the column containing product videos URLs (list)."""
    store_ix: str = "store"
    """Name of the column containing store/seller information."""
    categories_ix: str = "categories"
    """Name of the column containing category hierarchy (list)."""
    details_ix: str = "details"
    """Name of the column containing product details (dict)."""
    bought_together_ix: str = "bought_together"
    """Name of the column containing products bought together (list)."""

    @property
    def column_names(self) -> list[str]:
        return [
            self.main_category_ix,
            self.title_ix,
            self.average_rating_ix,
            self.rating_number_ix,
            self.features_ix,
            self.description_ix,
            self.price_ix,
            self.images_ix,
            self.videos_ix,
            self.store_ix,
            self.categories_ix,
            self.details_ix,
            self.item_ix,
            self.bought_together_ix,
        ]

    @property
    def dtype_dict(self) -> dict:
        return {
            self.main_category_ix: str,
            self.title_ix: str,
            self.average_rating_ix: "float32",
            self.rating_number_ix: "int64",
            self.features_ix: list,
            self.description_ix: list,
            self.price_ix: "float32",
            self.images_ix: list,
            self.videos_ix: list,
            self.store_ix: str,
            self.categories_ix: list,
            self.details_ix: dict,
            self.item_ix: str,
            self.bought_together_ix: list,
        }


@dataclass
class AmazonDigitalMusicItemMetadataConfig(AmazonItemMetadataConfig):
    """Amazon Digital Music metadata configuration."""

    remote_filename: str = "meta_Digital_Music.jsonl.gz"
    """Filename for Digital Music metadata."""

    dataset_url: str = (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
        "raw/meta_categories/meta_Digital_Music.jsonl.gz"
    )


@dataclass
class AmazonMoviesAndTVItemMetadataConfig(AmazonItemMetadataConfig):
    """Amazon Movies and TV metadata configuration."""

    remote_filename: str = "meta_Movies_and_TV.jsonl.gz"
    """Filename for Movies and TV metadata."""

    dataset_url: str = (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
        "raw/meta_categories/meta_Movies_and_TV.jsonl.gz"
    )


@dataclass
class AmazonSubscriptionBoxesItemMetadataConfig(AmazonItemMetadataConfig):
    """Amazon Subscription Boxes metadata configuration."""

    remote_filename: str = "meta_Subscription_Boxes.jsonl.gz"
    """Filename for Subscription Boxes metadata."""

    dataset_url: str = (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
        "raw/meta_categories/meta_Subscription_Boxes.jsonl.gz"
    )


@dataclass
class AmazonBooksItemMetadataConfig(AmazonItemMetadataConfig):
    """Amazon Books metadata configuration."""

    remote_filename: str = "meta_Books.jsonl.gz"
    """Filename for Books metadata."""

    dataset_url: str = (
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
        "raw/meta_categories/meta_Books.jsonl.gz"
    )
