import pandas as pd
import pytest

from streamsight.matrix import InteractionMatrix
from streamsight.settings import LeaveNOutSetting
from streamsight.settings.base import Setting


N_SEQ_DATA = 10
N = 1


@pytest.fixture
def setting(session_vars) -> LeaveNOutSetting:
    return LeaveNOutSetting(n_seq_data=N_SEQ_DATA, N=1, seed=session_vars["SEED"])


class TestLeaveNOut:
    def test_seed_value(self, setting: LeaveNOutSetting, session_vars: dict) -> None:
        assert setting.seed == session_vars["SEED"]

    def test_background_data(self, setting: Setting, matrix: InteractionMatrix) -> None:
        expected_background_data = pd.DataFrame(
            {
                "ts": [0, 1, 2, 4, 6, 7, 9, 10],
                "uid": [1, 2, 3, 2, 4, 3, 5, 5],
                "iid": [1, 1, 2, 2, 2, 1, 1, 2],
            }
        )

        setting.split(matrix)
        assert setting.background_data is not None
        actual_background_data = setting.background_data._df[["ts", "uid", "iid"]].reset_index(drop=True)
        assert actual_background_data.equals(expected_background_data)
