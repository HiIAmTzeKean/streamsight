import pandas as pd
import pytest

from streamsight.matrix import InteractionMatrix
from streamsight.settings.base import Setting
from streamsight.settings.single_time_point_setting import SingleTimePointSetting

BACKGROUND_T = 4
SEED = 42
N_SEQ_DATA=1

@pytest.fixture()
def setting():
    return SingleTimePointSetting(background_t=BACKGROUND_T,
                                  n_seq_data=N_SEQ_DATA,
                                  seed=SEED)

#TODO test when n_seq is set to 0
class TestSingleTimePointSetting():
    def test_seed_value(self, setting):
        assert setting.seed == SEED
    
    def test_empty_background_data(self, matrix:InteractionMatrix):
        setting_temp = SingleTimePointSetting(background_t=0,
                                  n_seq_data=N_SEQ_DATA,
                                  seed=SEED)
        assert setting_temp.t == 0
        with pytest.warns(UserWarning,
                          match="Background data resulting from SingleTimePointSetting is empty \(no interactions\)."
                          ):
            setting_temp.split(matrix)
        assert setting_temp.background_data is not None
        assert setting_temp.background_data._df.empty
        
    def test_no_n_seq_data(self, matrix:InteractionMatrix):
        setting_temp = SingleTimePointSetting(background_t=BACKGROUND_T,
                                  n_seq_data=0,
                                  seed=SEED)
        assert setting_temp.t == BACKGROUND_T
        setting_temp.split(matrix)
        assert setting_temp.unlabeled_data is not None
        assert setting_temp.unlabeled_data._df["iid"].nunique() == 1
        assert setting_temp.unlabeled_data._df["iid"].unique()[0] == -1
    
    def test_background_t_value(self, setting):
        assert setting.t == BACKGROUND_T
    
    def test_background_data(self, setting:Setting, matrix:InteractionMatrix):
        expected_background_data = matrix.timestamps_lt(BACKGROUND_T)
        
        setting.split(matrix)
        assert setting.background_data is not None
        assert setting.background_data._df.equals(expected_background_data._df)
    
    def test_ground_truth_data(self, setting:Setting, matrix:InteractionMatrix):
        expected_ground_truth = pd.DataFrame({
            "ts":   [4, 6, 7, 9],
            "uid":  [2, 4, 3, 5],
            "iid":  [2, 2, 1, 1]
        })
        setting.split(matrix)
        assert setting.ground_truth_data is not None
        actual_ground_truth = setting.ground_truth_data
        assert type(actual_ground_truth) is InteractionMatrix
        actual_ground_truth = actual_ground_truth._df[["ts","uid","iid"]].reset_index(drop=True)
        assert actual_ground_truth.equals(expected_ground_truth)

    def test_unlabeled_data(self, setting:Setting, matrix:InteractionMatrix):
        expected_unlabeled_data = pd.DataFrame({
            "ts":   [1, 2, 4, 6, 7, 9],
            "uid":   [2, 3, 2, 4, 3, 5],
            "iid":   [1, 2, -1, -1, -1, -1],
        })
        setting.split(matrix)
        assert setting.unlabeled_data is not None
        actual_unlabeled_data = setting.unlabeled_data
        assert type(actual_unlabeled_data) is InteractionMatrix
        actual_unlabeled_data = actual_unlabeled_data._df[["ts","uid","iid"]].reset_index(drop=True)
        assert actual_unlabeled_data.equals(expected_unlabeled_data)