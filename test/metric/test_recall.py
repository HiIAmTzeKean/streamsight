import pytest
import numpy as np
from streamsight.metrics import RecallK


@pytest.fixture()
def recall_100_timestamp() -> RecallK:
    return RecallK(100, 123)


@pytest.fixture()
def recall_100() -> RecallK:
    return RecallK(100)


@pytest.fixture()
def recall_default() -> RecallK:
    return RecallK()


class TestRecallK:
    def test_precision_name(self, recall_100_timestamp, recall_default):
        assert recall_100_timestamp.name == "RecallK_100"
        assert recall_default.name == "RecallK_10"

    def test_precision_k_values(
        self, recall_100_timestamp, recall_100, recall_default
    ):
        assert recall_100_timestamp.K == 100
        assert recall_100.K == 100
        assert recall_default.K == 10

    def test_precision_identifier(
        self, recall_100_timestamp, recall_100, recall_default
    ):
        assert (
            recall_100_timestamp.identifier
            == "RecallK(timestamp_limit=123,K=100)"
        )
        assert recall_100.identifier == "RecallK(K=100)"
        assert recall_default.identifier == "RecallK(K=10)"

    def test_precision_calculate(self, recall_default, X_pred, X_true):
        recall_default.calculate(X_true, X_pred)
        recall_default._scores
        assert pytest.approx(recall_default.macro_result, rel=0.01) == 0.83
        assert (
            pytest.approx(recall_default.micro_result["score"], rel=0.01) == np.array([1, 0.67])
        )
