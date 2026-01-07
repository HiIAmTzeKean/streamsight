import pytest
from scipy.sparse import csr_matrix


@pytest.fixture(scope="function")
def X_pred() -> csr_matrix:
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.23, 0.3, 0.5],
    )

    pred = csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(10, 5)
    )

    return pred


@pytest.fixture(scope="function")
def X_true() -> csr_matrix:
    true_users, true_items = [0, 0, 2, 2, 2], [0, 2, 0, 1, 3]

    true_data = csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(10, 5)
    )

    return true_data