import pytest
import numpy as np
from streamsight.metrics import PrecisionK


@pytest.fixture()
def precision_100_timestamp() -> PrecisionK:
    return PrecisionK(100, 123)


@pytest.fixture()
def precision_100() -> PrecisionK:
    return PrecisionK(100)


@pytest.fixture()
def precision_default() -> PrecisionK:
    return PrecisionK()


class TestPrecisionK:
    def test_precision_name(self, precision_100_timestamp, precision_default):
        assert precision_100_timestamp.name == "PrecisionK_100"
        assert precision_default.name == "PrecisionK_10"

    def test_precision_k_values(
        self, precision_100_timestamp, precision_100, precision_default
    ):
        assert precision_100_timestamp.K == 100
        assert precision_100.K == 100
        assert precision_default.K == 10

    def test_precision_identifier(
        self, precision_100_timestamp, precision_100, precision_default
    ):
        assert (
            precision_100_timestamp.identifier
            == "PrecisionK(timestamp_limit=123,K=100)"
        )
        assert precision_100.identifier == "PrecisionK(K=100)"
        assert precision_default.identifier == "PrecisionK(K=10)"

    def test_precision_calculate(self, precision_default, X_pred, X_true):
        precision_default.calculate(X_true, X_pred)
        precision_default._scores
        assert precision_default.macro_result == 0.2
        assert (
            pytest.approx(precision_default.micro_result["score"], rel=0.01) == np.array([0.2, 0.2])
        )
