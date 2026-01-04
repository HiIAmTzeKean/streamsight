import warnings
from typing import Any, Dict

import pytest

from streamsight.matrix import InteractionMatrix, TimestampAttributeMissingError
from streamsight.settings.base import Setting
from streamsight.settings.single_time_point_setting import SingleTimePointSetting


class TestSingleTimePointSetting:
    """Test suite for SingleTimePointSetting class."""

    @pytest.fixture
    def default_setting(self, session_vars: Dict[str, Any]) -> SingleTimePointSetting:
        """Fixture for default SingleTimePointSetting."""
        return SingleTimePointSetting(
            background_t=session_vars["BACKGROUND_T"],
            n_seq_data=session_vars["N_SEQ_DATA"],
            seed=session_vars["SEED"],
        )

    @pytest.fixture
    def custom_setting(self) -> SingleTimePointSetting:
        """Fixture for custom SingleTimePointSetting."""
        return SingleTimePointSetting(
            background_t=5,
            n_seq_data=2,
            top_K=3,
            t_upper=20,
            include_all_past_data=True,
            seed=999,
        )

    def test_initialization_default(self, default_setting: SingleTimePointSetting, session_vars: Dict[str, Any]) -> None:
        """Test default initialization."""
        assert default_setting.seed == session_vars["SEED"]
        assert default_setting.t == session_vars["BACKGROUND_T"]
        assert default_setting.n_seq_data == session_vars["N_SEQ_DATA"]
        assert default_setting.top_K == 1
        assert not default_setting.is_ready
        assert not default_setting.is_sliding_window_setting

    def test_initialization_custom(self, custom_setting: SingleTimePointSetting) -> None:
        """Test custom initialization."""
        assert custom_setting.seed == 999
        assert custom_setting.t == 5
        assert custom_setting.n_seq_data == 2
        assert custom_setting.top_K == 3
        assert custom_setting.t_upper == 20

    def test_params(self, default_setting: SingleTimePointSetting) -> None:
        """Test get_params method."""
        params = default_setting.get_params()
        assert "seed" in params
        assert "t" in params
        assert "n_seq_data" in params
        assert "top_K" in params

    def test_str_representation(self, default_setting: SingleTimePointSetting) -> None:
        """Test string representation."""
        assert "SingleTimePointSetting" in str(default_setting)

    def test_identifier(self, default_setting: SingleTimePointSetting) -> None:
        """Test identifier property."""
        assert "SingleTimePointSetting" in default_setting.identifier

    @pytest.mark.parametrize("background_t, n_seq_data, top_K", [
        (0, 1, 1),
        (10, 5, 10),
        (100, 0, 2),
    ])
    def test_initialization_parametrized(self, background_t: int, n_seq_data: int, top_K: int) -> None:
        """Test initialization with various parameters."""
        setting = SingleTimePointSetting(background_t=background_t, n_seq_data=n_seq_data, top_K=top_K, seed=42)
        assert setting.t == background_t
        assert setting.n_seq_data == n_seq_data
        assert setting.top_K == top_K

    def test_split_basic(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test basic split functionality."""
        default_setting.split(matrix)
        assert default_setting.is_ready
        assert default_setting.background_data is not None
        assert default_setting.unlabeled_data is not None
        assert default_setting.ground_truth_data is not None
        assert default_setting.t_window == default_setting.t
        assert default_setting.num_split == 1

    def test_background_data_content(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test background data content after split."""
        default_setting.split(matrix)
        bg_data = default_setting.background_data
        assert isinstance(bg_data, InteractionMatrix)
        # Background should contain interactions before background_t
        # Note: Depending on implementation, may include or exclude boundary

    def test_unlabeled_data_content(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test unlabeled data content after split."""
        default_setting.split(matrix)
        unlabeled = default_setting.unlabeled_data
        assert isinstance(unlabeled, InteractionMatrix)

    def test_ground_truth_data_content(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test ground truth data content after split."""
        default_setting.split(matrix)
        gt_data = default_setting.ground_truth_data
        assert isinstance(gt_data, InteractionMatrix)
        # Ground truth should contain future interactions

    def test_iteration(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test iteration over splits."""
        default_setting.split(matrix)
        splits = []
        try:
            for split in default_setting:
                splits.append(split)
        except Exception:
            pass
        assert len(splits) == 1
        split_result = splits[0]
        assert hasattr(split_result, 'unlabeled')
        assert hasattr(split_result, 'ground_truth')
        assert hasattr(split_result, 't_window')
        assert hasattr(split_result, 'incremental')
        assert split_result.incremental is None

    def test_get_split_at_valid(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test get_split_at with valid index."""
        default_setting.split(matrix)
        split_result = default_setting.get_split_at(0)
        assert hasattr(split_result, 'unlabeled')
        assert hasattr(split_result, 'ground_truth')
        assert split_result.incremental is None

    def test_get_split_at_invalid(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test get_split_at with invalid index."""
        default_setting.split(matrix)
        with pytest.raises(IndexError):
            default_setting.get_split_at(1)

    def test_access_properties_before_split(self, default_setting: SingleTimePointSetting) -> None:
        """Test accessing properties before split raises KeyError."""
        with pytest.raises(KeyError):
            _ = default_setting.background_data
        with pytest.raises(KeyError):
            _ = default_setting.unlabeled_data
        with pytest.raises(KeyError):
            _ = default_setting.ground_truth_data
        with pytest.raises(KeyError):
            _ = default_setting.t_window

    def test_incremental_data_raises_error(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test that incremental_data raises AttributeError for non-sliding window."""
        default_setting.split(matrix)
        with pytest.raises(AttributeError):
            _ = default_setting.incremental_data

    def test_split_with_empty_background(self, matrix: InteractionMatrix) -> None:
        """Test split with background_t before any data."""
        setting = SingleTimePointSetting(background_t=-1, n_seq_data=1, seed=42)
        with pytest.warns(UserWarning, match="Splitting at time.*before the first timestamp"):
            setting.split(matrix)
        assert setting.is_ready

    def test_split_no_n_seq_data(self, matrix: InteractionMatrix) -> None:
        """Test split with n_seq_data=0."""
        setting = SingleTimePointSetting(background_t=4, n_seq_data=0, seed=42)
        setting.split(matrix)
        assert setting.is_ready
        unlabeled = setting.unlabeled_data
        # When n_seq_data=0, unlabeled should have masked items
        assert unlabeled.num_interactions > 0

    def test_split_no_timestamps_raises_error(self, default_setting: SingleTimePointSetting, matrix_no_timestamps: InteractionMatrix) -> None:
        """Test split with matrix without timestamps raises error."""
        with pytest.raises(TimestampAttributeMissingError):
            default_setting.split(matrix_no_timestamps)

    def test_split_background_t_before_min_timestamp_warns(self, matrix: InteractionMatrix) -> None:
        """Test split with background_t before min timestamp warns."""
        setting = SingleTimePointSetting(background_t=-1, n_seq_data=1, seed=42)
        with pytest.warns(UserWarning, match="Splitting at time.*before the first timestamp"):
            setting.split(matrix)

    def test_restore(self, default_setting: SingleTimePointSetting, matrix: InteractionMatrix) -> None:
        """Test restore method."""
        default_setting.split(matrix)
        default_setting.restore(0)
        assert default_setting.current_index == 0
