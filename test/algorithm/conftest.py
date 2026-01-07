import pytest
from streamsight.settings.sliding_window_setting import SlidingWindowSetting
from test.conftest import BACKGROUND_T, WINDOW_SIZE, SEED, N_SEQ_DATA, SEED

@pytest.fixture()
def setting(test_dataset):
    data = test_dataset.load()
    setting_obj = SlidingWindowSetting(background_t=BACKGROUND_T,
                                window_size=WINDOW_SIZE,
                                n_seq_data=N_SEQ_DATA,
                                seed=SEED)
    setting_obj.split(data)
    return setting_obj