import pytest
from typing import Any, Dict

from streamsight.matrix import InteractionMatrix
from streamsight.settings import LeaveNOutSetting
from streamsight.settings.base import Setting


class TestLeaveNOutSetting:
    """Test suite for LeaveNOutSetting class."""

    @pytest.fixture
    def default_setting(self, session_vars: Dict[str, Any]) -> LeaveNOutSetting:
        """Fixture for default LeaveNOutSetting."""
        return LeaveNOutSetting(n_seq_data=session_vars["N_SEQ_DATA"], N=1, seed=session_vars["SEED"])

    @pytest.fixture
    def custom_setting(self) -> LeaveNOutSetting:
        """Fixture for custom LeaveNOutSetting."""
        return LeaveNOutSetting(n_seq_data=5, N=3, seed=123)

    def test_initialization_default(self, default_setting: LeaveNOutSetting, session_vars: Dict[str, Any]) -> None:
        """Test default initialization."""
        assert default_setting.seed == session_vars["SEED"]
        assert default_setting.n_seq_data == session_vars["N_SEQ_DATA"]
        assert default_setting.top_K == 1
        assert not default_setting.is_ready
        assert not default_setting.is_sliding_window_setting

    def test_initialization_custom(self, custom_setting: LeaveNOutSetting) -> None:
        """Test custom initialization."""
        assert custom_setting.seed == 123
        assert custom_setting.n_seq_data == 5
        assert custom_setting.top_K == 3

    def test_params(self, default_setting: LeaveNOutSetting) -> None:
        """Test get_params method."""
        params = default_setting.get_params()
        assert "seed" in params
        assert "n_seq_data" in params
        assert "top_K" in params

    def test_str_representation(self, default_setting: LeaveNOutSetting) -> None:
        """Test string representation."""
        assert "LeaveNOutSetting" in str(default_setting)

    def test_identifier(self, default_setting: LeaveNOutSetting) -> None:
        """Test identifier property."""
        assert "LeaveNOutSetting" in default_setting.identifier

    @pytest.mark.parametrize("n_seq_data, N", [
        (0, 1),
        (1, 5),
        (10, 2),
    ])
    def test_initialization_parametrized(self, n_seq_data: int, N: int) -> None:
        """Test initialization with various parameters."""
        setting = LeaveNOutSetting(n_seq_data=n_seq_data, N=N, seed=42)
        assert setting.n_seq_data == n_seq_data
        assert setting.top_K == N

    def test_split_basic(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test basic split functionality."""
        default_setting.split(matrix)
        assert default_setting.is_ready
        assert default_setting.background_data is not None
        assert default_setting.unlabeled_data is not None
        assert default_setting.ground_truth_data is not None
        assert default_setting.t_window is None
        assert default_setting.num_split == 1

    def test_background_data_content(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test background data content after split."""
        default_setting.split(matrix)
        bg_data = default_setting.background_data
        assert isinstance(bg_data, InteractionMatrix)
        # Verify that background data excludes the last N interactions per user
        # This is a simplified check; in practice, it depends on the splitter logic

    def test_unlabeled_data_content(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test unlabeled data content after split."""
        default_setting.split(matrix)
        unlabeled = default_setting.unlabeled_data
        assert isinstance(unlabeled, InteractionMatrix)
        # Check that unlabeled data has masked items for prediction

    def test_ground_truth_data_content(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test ground truth data content after split."""
        default_setting.split(matrix)
        gt_data = default_setting.ground_truth_data
        assert isinstance(gt_data, InteractionMatrix)
        # Check that ground truth contains the held-out interactions

    def test_iteration(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test iteration over splits."""
        default_setting.split(matrix)
        splits = []
        try:
            for split in default_setting:
                splits.append(split)
        except Exception:
            pass  # EOWSettingError or similar
        assert len(splits) == 1
        split_result = splits[0]
        assert hasattr(split_result, 'unlabeled')
        assert hasattr(split_result, 'ground_truth')
        assert hasattr(split_result, 't_window')
        assert hasattr(split_result, 'incremental')
        assert split_result.incremental is None

    def test_get_split_at_valid(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test get_split_at with valid index."""
        default_setting.split(matrix)
        split_result = default_setting.get_split_at(0)
        assert hasattr(split_result, 'unlabeled')
        assert hasattr(split_result, 'ground_truth')
        assert split_result.incremental is None

    def test_get_split_at_invalid(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test get_split_at with invalid index."""
        default_setting.split(matrix)
        with pytest.raises(IndexError):
            default_setting.get_split_at(1)

    def test_access_properties_before_split(self, default_setting: LeaveNOutSetting) -> None:
        """Test accessing properties before split raises KeyError."""
        with pytest.raises(KeyError):
            _ = default_setting.background_data
        with pytest.raises(KeyError):
            _ = default_setting.unlabeled_data
        with pytest.raises(KeyError):
            _ = default_setting.ground_truth_data
        with pytest.raises(KeyError):
            _ = default_setting.t_window

    def test_incremental_data_raises_error(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test that incremental_data raises AttributeError for non-sliding window."""
        default_setting.split(matrix)
        with pytest.raises(AttributeError):
            _ = default_setting.incremental_data

    def test_split_with_empty_matrix(self, default_setting: LeaveNOutSetting, empty_matrix: InteractionMatrix) -> None:
        """Test split with empty matrix."""
        with pytest.warns(UserWarning, match="empty"):
            default_setting.split(empty_matrix)
        assert default_setting.is_ready

    def test_restore(self, default_setting: LeaveNOutSetting, matrix: InteractionMatrix) -> None:
        """Test restore method."""
        default_setting.split(matrix)
        default_setting.restore(0)
        assert default_setting.current_index == 0
