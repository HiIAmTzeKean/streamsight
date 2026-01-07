import pytest

from streamsight.matrix import InteractionMatrix
from streamsight.settings.base import Setting
from streamsight.settings.sliding_window_setting import SlidingWindowSetting

BACKGROUND_T = 4
WINDOW_SIZE = 3
SEED = 42
N_SEQ_DATA=1

@pytest.fixture()
def setting():
    return SlidingWindowSetting(background_t=BACKGROUND_T,
                                window_size=WINDOW_SIZE,
                                n_seq_data=N_SEQ_DATA,
                                seed=SEED)


class TestSlidingWindowSetting():
    def test_seed_value(self, setting):
        assert setting.seed == SEED

# TODO test if the overall global user and item id are correctly masked
    def test_background_t_value(self, setting):
        assert setting.t == BACKGROUND_T
        
    def test_split_properties(self, setting: Setting,  matrix: InteractionMatrix):
        setting._split(matrix)
        assert setting._num_split_set == 3
        
    def test_access_split_attributes_before_split(self, setting: Setting):
        with pytest.raises(KeyError) as e_info:
            setting.background_data
        with pytest.raises(KeyError) as e_info:
            setting.unlabeled_data
        with pytest.raises(KeyError) as e_info:
            setting.ground_truth_data
        with pytest.raises(KeyError) as e_info:
            setting.incremental_data
            
    def test_background_data(self, setting: Setting, matrix: InteractionMatrix):
        expected_background_data = matrix.timestamps_lt(BACKGROUND_T)

        setting.split(matrix)
        assert setting.background_data is not None
        assert setting.background_data._df.equals(
            expected_background_data._df)

    def test_incremental_data(self, setting: Setting, matrix: InteractionMatrix):
        expected_incremental_data_0 = matrix.timestamps_gte(BACKGROUND_T).timestamps_lt(BACKGROUND_T + WINDOW_SIZE)
        expected_incremental_data_1 = matrix.timestamps_gte(BACKGROUND_T + WINDOW_SIZE).timestamps_lt(BACKGROUND_T + 2*WINDOW_SIZE)
        setting.split(matrix)
        assert setting.incremental_data[0]._df.equals(
            expected_incremental_data_0._df)
        assert setting.incremental_data[1]._df.equals(
            expected_incremental_data_1._df)
        
    def test_t_window(self, setting: Setting, matrix: InteractionMatrix):
        setting.split(matrix)
        assert setting.t_window == [4, 7, 10]
