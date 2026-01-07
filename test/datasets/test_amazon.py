import os
import warnings

import pandas as pd
import pytest

from streamsight.datasets import AmazonMusicDataset, Dataset
from streamsight.matrix import InteractionMatrix
from streamsight.preprocessing import MinItemsPerUser


@pytest.fixture()
def dataset_path(filename):
    return os.path.join(os.getcwd(),"test","data")

@pytest.fixture()
def filename():
    return "amazon_digitalmusic_dataset.csv"

@pytest.fixture()
def dataset(dataset_path, filename) -> Dataset:
    return AmazonMusicDataset(
            base_path=dataset_path, filename=filename)

class TestAmazonMusicDataset:
    def test_amazon_music_load_dataframe(self, dataset):
        df = dataset._load_dataframe()
        assert type(df) == pd.DataFrame
        assert (df.columns == [dataset.ITEM_IX, dataset.USER_IX,
                dataset.RATING_IX, dataset.TIMESTAMP_IX]).all()

    def test_amazon_music_load_with_filter(self, dataset):
        """Test the load method of the AmazonMusicDataset class."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # ensure no warning is raised for loading with filter
            data = dataset.load()
        assert type(data) == InteractionMatrix
        
    def test_amazon_music_load_without_filter(self, dataset):
        """Test the load method of the AmazonMusicDataset class."""
        with pytest.warns(UserWarning):
            data = dataset.load(apply_filters=False)
        assert type(data) == InteractionMatrix
    
    def test_amazon_music_add_filter(self, dataset):
        """Test the add_filter method of the AmazonMusicDataset class."""
        data = dataset.add_filter(
            MinItemsPerUser(2, dataset.ITEM_IX, dataset.USER_IX)
        )
        data = dataset.load()
        
        