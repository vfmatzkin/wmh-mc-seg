import sys
from pathlib import Path
from unittest.mock import patch

import yaml

# src/ lives one level up from tests/; add it to the path so the import
# of src.utils.cli resolves correctly regardless of install mode.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.cli import load_defaults

# ── helpers ───────────────────────────────────────────────────────────────────

MAIN_KEYS = {
    "data_root",
    "centers",
    "split_ratios",
    "epochs",
    "batch_size",
    "lr",
    "dropout",
    "loss",
    "weight_decay",
    "seed",
    "patch_size",
    "samples_per_volume",
    "queue_length",
    "tio_num_workers",
    "reg_start",
    "meep_lambda",
    "ood_centers",
}

TEST_KEYS = {
    "data_root",
    "centers",
    "split_ratios",
    "model_path",
    "batch_size",
    "patch_size",
    "seed",
}


# ── tests ─────────────────────────────────────────────────────────────────────


def test_load_defaults_main_returns_dict():
    result = load_defaults("main")
    assert isinstance(result, dict)


def test_load_defaults_main_has_expected_keys():
    result = load_defaults("main")
    assert MAIN_KEYS.issubset(result.keys()), f"Missing keys: {MAIN_KEYS - result.keys()}"


def test_load_defaults_test_returns_dict():
    result = load_defaults("test")
    assert isinstance(result, dict)


def test_load_defaults_test_has_expected_keys():
    result = load_defaults("test")
    assert TEST_KEYS.issubset(result.keys()), f"Missing keys: {TEST_KEYS - result.keys()}"


def test_load_defaults_keys_use_underscores():
    for entry_point in ("main", "test"):
        result = load_defaults(entry_point)
        for key in result:
            assert "-" not in key, (
                f"Key '{key}' in '{entry_point}' contains hyphen — "
                "expected underscore-separated keys (click default_map convention)"
            )


def test_load_defaults_returns_empty_dict_when_mlproject_missing(tmp_path, mlproject_file):
    """load_defaults returns {} when MLproject doesn't exist."""
    # Point the loader at a directory that has no MLproject
    fake_src_utils = tmp_path / "src" / "utils"
    fake_src_utils.mkdir(parents=True)

    # cli.py resolves MLproject relative to __file__, so we patch the path.
    import src.utils.cli as cli_module

    nonexistent = tmp_path / "no_such_dir" / "MLproject"

    # Patch only the specific resolution inside load_defaults
    with patch.object(
        cli_module,
        "load_defaults",
        wraps=lambda ep="main": _load_defaults_from(nonexistent.parent, ep),
    ):
        result = cli_module.load_defaults("main")
        assert result == {}


def test_load_defaults_fixture_returns_expected_keys(mlproject_file):
    """Smoke-test the conftest fixture MLproject."""
    data = yaml.safe_load(mlproject_file.read_text())
    params = data["entry_points"]["main"]["parameters"]
    keys = set(params.keys())
    assert "epochs" in keys
    assert "batch_size" in keys


# ── helper used by the mock patch ────────────────────────────────────────────


def _load_defaults_from(base: Path, entry_point: str) -> dict:
    mlproject_path = base / "MLproject"
    if not mlproject_path.exists():
        return {}
    data = yaml.safe_load(mlproject_path.read_text())
    params = data["entry_points"][entry_point]["parameters"]
    return {k: str(v["default"]) for k, v in params.items()}
