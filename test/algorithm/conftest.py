import pytest

from streamsight.datasets.base import Dataset
from streamsight.settings.sliding_window_setting import SlidingWindowSetting


@pytest.fixture
def setting(test_dataset: Dataset, session_vars: dict) -> SlidingWindowSetting:
    data = test_dataset.load()
    setting_obj = SlidingWindowSetting(
        background_t=session_vars["BACKGROUND_T"],
        window_size=session_vars["WINDOW_SIZE"],
        n_seq_data=session_vars["N_SEQ_DATA"],
        seed=session_vars["SEED"],
    )
    setting_obj.split(data)
    return setting_obj
