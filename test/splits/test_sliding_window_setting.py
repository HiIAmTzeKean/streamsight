from typing import Any, Dict, List

import pytest

from streamsight.matrix import InteractionMatrix, TimestampAttributeMissingError
from streamsight.settings.base import Setting
from streamsight.settings.sliding_window_setting import SlidingWindowSetting


class TestSlidingWindowSetting:
    """Test suite for SlidingWindowSetting class."""

    @pytest.fixture
    def default_setting(self, session_vars: Dict[str, Any]) -> SlidingWindowSetting:
        """Fixture for default SlidingWindowSetting."""
        return SlidingWindowSetting(
            background_t=session_vars["BACKGROUND_T"],
            window_size=session_vars["WINDOW_SIZE"],
            n_seq_data=session_vars["N_SEQ_DATA"],
            seed=session_vars["SEED"],
        )

    @pytest.fixture
    def custom_setting(self) -> SlidingWindowSetting:
        """Fixture for custom SlidingWindowSetting."""
        return SlidingWindowSetting(
            background_t=5,
            window_size=2,
            n_seq_data=3,
            top_K=5,
            t_upper=15,
            t_ground_truth_window=3,
            seed=777,
        )

    def test_initialization_default(self, default_setting: SlidingWindowSetting, session_vars: Dict[str, Any]) -> None:
        """Test default initialization."""
        assert default_setting.seed == session_vars["SEED"]
        assert default_setting.t == session_vars["BACKGROUND_T"]
        assert default_setting.window_size == session_vars["WINDOW_SIZE"]
        assert default_setting.n_seq_data == session_vars["N_SEQ_DATA"]
        assert default_setting.top_K == 10  # default
        assert not default_setting.is_ready
        assert default_setting.is_sliding_window_setting

    def test_initialization_custom(self, custom_setting: SlidingWindowSetting) -> None:
        """Test custom initialization."""
        assert custom_setting.seed == 777
        assert custom_setting.t == 5
        assert custom_setting.window_size == 2
        assert custom_setting.n_seq_data == 3
        assert custom_setting.top_K == 5
        assert custom_setting.t_upper == 15
        assert custom_setting.t_ground_truth_window == 3

    def test_params(self, default_setting: SlidingWindowSetting) -> None:
        """Test get_params method."""
        params = default_setting.get_params()
        assert "seed" in params
        assert "t" in params
        assert "window_size" in params
        assert "n_seq_data" in params
        assert "top_K" in params

    def test_str_representation(self, default_setting: SlidingWindowSetting) -> None:
        """Test string representation."""
        assert "SlidingWindowSetting" in str(default_setting)

    def test_identifier(self, default_setting: SlidingWindowSetting) -> None:
        """Test identifier property."""
        assert "SlidingWindowSetting" in default_setting.identifier

    @pytest.mark.parametrize("background_t, window_size, n_seq_data", [
        (0, 1, 0),
        (10, 5, 2),
        (50, 10, 1),
    ])
    def test_initialization_parametrized(self, background_t: int, window_size: int, n_seq_data: int) -> None:
        """Test initialization with various parameters."""
        setting = SlidingWindowSetting(background_t=background_t, window_size=window_size, n_seq_data=n_seq_data, seed=42)
        assert setting.t == background_t
        assert setting.window_size == window_size
        assert setting.n_seq_data == n_seq_data

    def test_invalid_t_upper_raises_error(self) -> None:
        """Test that t_upper < background_t raises ValueError."""
        with pytest.raises(ValueError, match="t_upper must be greater than background_t"):
            SlidingWindowSetting(background_t=10, t_upper=5, seed=42)

    def test_split_basic(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test basic split functionality."""
        default_setting.split(matrix)
        assert default_setting.is_ready
        assert default_setting.background_data is not None
        assert isinstance(default_setting.unlabeled_data, list)
        assert isinstance(default_setting.ground_truth_data, list)
        assert isinstance(default_setting.incremental_data, list)
        assert isinstance(default_setting.t_window, list)
        assert default_setting.num_split > 1

    def test_background_data_content(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test background data content after split."""
        default_setting.split(matrix)
        bg_data = default_setting.background_data
        assert isinstance(bg_data, InteractionMatrix)
        # Background should contain interactions before background_t

    def test_unlabeled_data_content(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test unlabeled data content after split."""
        default_setting.split(matrix)
        unlabeled_list = default_setting.unlabeled_data
        assert isinstance(unlabeled_list, list)
        assert len(unlabeled_list) == default_setting.num_split
        for unlabeled in unlabeled_list:
            assert isinstance(unlabeled, InteractionMatrix)

    def test_ground_truth_data_content(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test ground truth data content after split."""
        default_setting.split(matrix)
        gt_list = default_setting.ground_truth_data
        assert isinstance(gt_list, list)
        assert len(gt_list) == default_setting.num_split
        for gt in gt_list:
            assert isinstance(gt, InteractionMatrix)

    def test_incremental_data_content(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test incremental data content after split."""
        default_setting.split(matrix)
        inc_list = default_setting.incremental_data
        assert isinstance(inc_list, list)
        assert len(inc_list) == default_setting.num_split
        for inc in inc_list:
            assert isinstance(inc, InteractionMatrix)

    def test_t_window_content(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test t_window content after split."""
        default_setting.split(matrix)
        t_window_list = default_setting.t_window
        assert isinstance(t_window_list, list)
        assert len(t_window_list) == default_setting.num_split
        # Check that t_windows are increasing
        assert all(t_window_list[i] <= t_window_list[i+1] for i in range(len(t_window_list)-1))

    def test_iteration(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test iteration over splits."""
        default_setting.split(matrix)
        splits = []
        try:
            for split in default_setting:
                splits.append(split)
        except Exception:
            pass
        assert len(splits) == default_setting.num_split
        for split_result in splits:
            assert hasattr(split_result, 'unlabeled')
            assert hasattr(split_result, 'ground_truth')
            assert hasattr(split_result, 't_window')
            assert hasattr(split_result, 'incremental')
            # Incremental may be None for some splits

    def test_get_split_at_valid(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test get_split_at with valid index."""
        default_setting.split(matrix)
        for i in range(default_setting.num_split):
            split_result = default_setting.get_split_at(i)
            assert hasattr(split_result, 'unlabeled')
            assert hasattr(split_result, 'ground_truth')
            assert hasattr(split_result, 'incremental')
            # Incremental may be None for some splits

    def test_get_split_at_invalid(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test get_split_at with invalid index."""
        default_setting.split(matrix)
        with pytest.raises(IndexError):
            default_setting.get_split_at(default_setting.num_split)

    def test_access_properties_before_split(self, default_setting: SlidingWindowSetting) -> None:
        """Test accessing properties before split raises KeyError."""
        with pytest.raises(KeyError):
            _ = default_setting.background_data
        with pytest.raises(KeyError):
            _ = default_setting.unlabeled_data
        with pytest.raises(KeyError):
            _ = default_setting.ground_truth_data
        with pytest.raises(KeyError):
            _ = default_setting.incremental_data
        with pytest.raises(KeyError):
            _ = default_setting.t_window

    def test_split_no_timestamps_raises_error(self, default_setting: SlidingWindowSetting, matrix_no_timestamps: InteractionMatrix) -> None:
        """Test split with matrix without timestamps raises error."""
        with pytest.raises(TimestampAttributeMissingError):
            default_setting.split(matrix_no_timestamps)

    def test_split_background_t_before_min_timestamp_warns(self, matrix: InteractionMatrix) -> None:
        """Test split with background_t before min timestamp warns."""
        setting = SlidingWindowSetting(background_t=-1, window_size=3, seed=42)
        with pytest.warns(UserWarning, match="Splitting at time.*before the first"):
            setting.split(matrix)

    def test_restore(self, default_setting: SlidingWindowSetting, matrix: InteractionMatrix) -> None:
        """Test restore method."""
        default_setting.split(matrix)
        default_setting.restore(0)
        assert default_setting.current_index == 0

    def test_split_with_small_window(self, matrix: InteractionMatrix) -> None:
        """Test split with small window size."""
        setting = SlidingWindowSetting(background_t=4, window_size=1, n_seq_data=0, seed=42)
        setting.split(matrix)
        assert setting.num_split > 1  # Should create multiple splits
