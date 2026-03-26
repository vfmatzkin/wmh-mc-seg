"""Tests for src/analysis.py — cache helpers and build_plot_data."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module-level stub setup
#
# src/analysis.py has several heavy top-level imports that are not available
# in the test environment (calibration, nibabel, SimpleITK, medpy, src.plot).
# We inject stubs into sys.modules before importing the module under test so
# the import succeeds without any real data dependencies.
# ---------------------------------------------------------------------------


def _build_stubs() -> dict:
    """Return a mapping of module-name -> stub to inject into sys.modules."""
    stubs: dict[str, object] = {}

    # calibration
    cal = ModuleType("calibration")
    cal.get_ece = MagicMock(return_value=0.0)  # type: ignore[attr-defined]
    stubs["calibration"] = cal

    # nibabel
    nib = ModuleType("nibabel")
    nib.load = MagicMock()  # type: ignore[attr-defined]
    stubs["nibabel"] = nib

    # SimpleITK
    sitk = ModuleType("SimpleITK")
    stubs["SimpleITK"] = sitk

    # medpy and sub-packages
    medpy = ModuleType("medpy")
    medpy_metric = ModuleType("medpy.metric")
    medpy_binary = ModuleType("medpy.metric.binary")
    medpy_binary.dc = MagicMock(return_value=1.0)  # type: ignore[attr-defined]
    medpy.metric = medpy_metric  # type: ignore[attr-defined]
    medpy_metric.binary = medpy_binary  # type: ignore[attr-defined]
    stubs["medpy"] = medpy
    stubs["medpy.metric"] = medpy_metric
    stubs["medpy.metric.binary"] = medpy_binary

    # src.plot
    src_plot = ModuleType("src.plot")
    src_plot.entropy = MagicMock(return_value=0.0)  # type: ignore[attr-defined]
    src_plot.get_b_mask_path = MagicMock(return_value="")  # type: ignore[attr-defined]
    src_plot.get_array_from_nifti = MagicMock()  # type: ignore[attr-defined]
    stubs["src.plot"] = src_plot

    # sklearn.calibration
    skl_cal = ModuleType("sklearn.calibration")
    skl_cal.calibration_curve = MagicMock(return_value=([], []))  # type: ignore[attr-defined]
    stubs["sklearn.calibration"] = skl_cal

    return stubs


@pytest.fixture(scope="module")
def analysis_module():
    """Import src.analysis with all heavy dependencies stubbed out."""
    stubs = _build_stubs()
    with patch.dict(sys.modules, stubs):
        # Force a fresh import (in case it was already cached without stubs)
        sys.modules.pop("src.analysis", None)
        import src.analysis as mod

        yield mod
        sys.modules.pop("src.analysis", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_dm(splits=None):
    """Return a mock WMHDataModule whose generate_splits() returns splits."""
    if splits is None:
        splits = []
    dm = MagicMock()
    dm.generate_splits.return_value = ([], [], splits)
    return dm


# ---------------------------------------------------------------------------
# 2.1  build_plot_data returns dict with required keys
# ---------------------------------------------------------------------------


def test_build_plot_data_returns_required_keys(tmp_path, analysis_module):
    fake_root = str(tmp_path)
    fake_dm = _make_fake_dm([])
    fake_datamodules = MagicMock(WMHDataModule=MagicMock(return_value=fake_dm))

    with patch.dict(sys.modules, {"src.datamodules": fake_datamodules}):
        result = analysis_module.build_plot_data(fake_root)

    assert isinstance(result, dict)
    for key in ("centers_train", "runs_to_compare", "centers_test", "test_splits", "losses"):
        assert key in result, f"Missing key: {key}"

    assert result["runs_to_compare"] == {}
    assert isinstance(result["centers_train"], list)
    assert isinstance(result["losses"], list)
    assert isinstance(result["test_splits"], dict)
    assert isinstance(result["centers_test"], list)


# ---------------------------------------------------------------------------
# 2.2  build_plot_data raises FileNotFoundError for non-existent data_root
# ---------------------------------------------------------------------------


def test_build_plot_data_raises_for_missing_data_root(analysis_module):
    with pytest.raises(FileNotFoundError):
        analysis_module.build_plot_data("/nonexistent/path/that/does/not/exist")


# ---------------------------------------------------------------------------
# 2.3  cache-miss: compute function is called and CSV is written
# ---------------------------------------------------------------------------


def test_cached_compute_cache_miss_writes_csv(tmp_path, analysis_module):
    cache_file = tmp_path / "output.csv"
    expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    compute_fn = MagicMock(return_value=expected)

    result = analysis_module._cached_compute(compute_fn, cache_path=str(cache_file), x=1)

    compute_fn.assert_called_once_with(x=1)
    assert cache_file.exists(), "CSV should have been written on cache miss"
    written = pd.read_csv(cache_file)
    pd.testing.assert_frame_equal(written, expected)
    pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 2.4  cache-hit: CSV is read and compute function is NOT called
# ---------------------------------------------------------------------------


def test_cached_compute_cache_hit_returns_cached_data(tmp_path, analysis_module):
    cache_file = tmp_path / "cached.csv"
    cached_df = pd.DataFrame({"x": [10, 20]})
    cached_df.to_csv(cache_file, index=False)

    compute_fn = MagicMock()

    result = analysis_module._cached_compute(compute_fn, cache_path=str(cache_file))

    compute_fn.assert_not_called()
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        cached_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 2.5  cache_path=None: compute is called, no CSV written
# ---------------------------------------------------------------------------


def test_cached_compute_no_cache_path_returns_result(tmp_path, analysis_module):
    expected = pd.DataFrame({"val": [99]})
    compute_fn = MagicMock(return_value=expected)

    result = analysis_module._cached_compute(compute_fn, cache_path=None, val=99)

    compute_fn.assert_called_once_with(val=99)
    assert list(tmp_path.iterdir()) == [], "No file should be written when cache_path is None"
    pd.testing.assert_frame_equal(result, expected)
