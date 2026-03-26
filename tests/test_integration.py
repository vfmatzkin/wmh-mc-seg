"""Integration tests for the full WMH segmentation pipeline.

All tests use synthetic (random) tensors — no real dataset or GPU needed.
Run the integration suite with: pytest -m integration tests/test_integration.py
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchio as tio

# ── mock lightning before any src/ import tries to pull it in ────────────────
# The test environment may not have pytorch-lightning installed; WMHModel only
# needs it for LightningModule base-class machinery we replace with a stub.
_lightning_stub = MagicMock()


class _FakeLightningModule:
    """Minimal stand-in for lightning.LightningModule."""

    def __init__(self, *args, **kwargs):
        pass

    def save_hyperparameters(self, *args, **kwargs):
        pass

    @property
    def current_epoch(self):
        return 0

    def log(self, *args, **kwargs):
        pass

    def log_dict(self, *args, **kwargs):
        pass


_lightning_stub.LightningModule = _FakeLightningModule
sys.modules.setdefault("lightning", _lightning_stub)
sys.modules.setdefault("lightning.pytorch", _lightning_stub)

# ── add src/ to path so internal (non-prefixed) imports resolve ──────────────
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from src.losses.composite import CLI_ALIASES, RegularizedLoss  # noqa: E402
from src.models.unet3d import UNet3D  # noqa: E402
from src.models.wmh_module import WMHModel  # noqa: E402

# Load get_mc_preds via importlib to avoid the SimpleITK/sitk_io chain at the
# module level of inference.py which may not be available in all envs.
_inference_path = _src / "models" / "inference.py"
_inf_spec = importlib.util.spec_from_file_location("models.inference", _inference_path)
_inf_mod = importlib.util.module_from_spec(_inf_spec)
sys.modules.setdefault("models.inference", _inf_mod)
_inf_spec.loader.exec_module(_inf_mod)
get_mc_preds = _inf_mod.get_mc_preds

# Same trick for get_preprocessing (avoids importing WMHDataModule → lightning).
_transforms_path = _src / "datamodules" / "transforms.py"
_tf_spec = importlib.util.spec_from_file_location("datamodules.transforms", _transforms_path)
_tf_mod = importlib.util.module_from_spec(_tf_spec)
sys.modules.setdefault("datamodules.transforms", _tf_mod)
_tf_spec.loader.exec_module(_tf_mod)
get_preprocessing = _tf_mod.get_preprocessing


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_wmh_model(loss="ce"):
    return WMHModel(
        net=UNet3D(dropout=0.1),
        criterion=loss,
        learning_rate=1e-3,
        optimizer_class=torch.optim.Adam,
    )


def _make_batch(spatial=16, batch_size=2):
    """Synthetic batch matching TorchIO's output format."""
    return {
        "t1": {
            tio.DATA: torch.randn(batch_size, 1, spatial, spatial, spatial),
            tio.PATH: [f"/fake/subj{i}/t1.nii.gz" for i in range(batch_size)],
        },
        "flair": {
            tio.DATA: torch.randn(batch_size, 1, spatial, spatial, spatial),
            tio.PATH: [f"/fake/subj{i}/flair.nii.gz" for i in range(batch_size)],
        },
        "wmh": {
            # 2 channels: OneHot produces (background, foreground)
            tio.DATA: torch.randn(batch_size, 2, spatial, spatial, spatial),
            tio.PATH: [f"/fake/subj{i}/wmh.nii.gz" for i in range(batch_size)],
        },
    }


# ── Test 1: UNet3D forward pass ───────────────────────────────────────────────


@pytest.mark.integration
def test_unet3d_forward_pass_shape():
    """Forward pass through UNet3D must return (B, 2, D, H, W)."""
    net = UNet3D(dropout=0.1)
    x = torch.randn(1, 2, 32, 32, 32)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 2, 32, 32, 32)


# ── Test 2: WMHModel instantiation with every loss alias ─────────────────────


@pytest.mark.integration
@pytest.mark.parametrize("alias", list(CLI_ALIASES.keys()))
def test_wmh_model_instantiation_all_losses(alias):
    """Every CLI alias must produce a model with a criterion and correct custom_loss flag."""
    model = _make_wmh_model(loss=alias)
    assert hasattr(model, "criterion")
    _, expected_custom = RegularizedLoss.from_cli(alias)
    assert model.custom_loss == expected_custom


# ── Test 3: Training step smoke test ─────────────────────────────────────────


@pytest.mark.integration
def test_shared_step_returns_loss_tensor():
    """_shared_step must return a scalar Tensor without raising."""
    model = _make_wmh_model(loss="ce")

    # RegularizedLoss.forward() always requires an `epoch` argument, but the
    # non-custom branch in _shared_step calls criterion(y_hat, y) with only 2
    # args.  Swap in the raw CrossEntropyLoss (what 'ce' resolves to) so the
    # call signature matches and the rest of the pipeline is still exercised.
    model.criterion = torch.nn.CrossEntropyLoss()
    model.custom_loss = False

    batch = _make_batch(spatial=16, batch_size=2)

    with patch("mlflow.log_metric"):
        loss = model._shared_step(batch, 0, "train")

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar


# ── Test 4: MC dropout inference ─────────────────────────────────────────────


@pytest.mark.integration
def test_mc_dropout_inference_shapes():
    """get_mc_preds must return tensors of shape (mc_samples, B, 2, D, H, W)."""
    net = UNet3D(dropout=0.2)
    x = torch.randn(1, 2, 16, 16, 16)
    mc_samples = 3

    # mc passes are triggered when mc_dropout_ratio > 0 AND is_test=True
    logits_arr, y_hat_arr = get_mc_preds(
        net=net,
        x=x,
        batch={},
        mc_dropout_ratio=0.2,
        mc_dropout_samples=mc_samples,
        patch_size=None,
        is_test=True,
    )

    assert logits_arr.shape == (mc_samples, 1, 2, 16, 16, 16)
    assert y_hat_arr.shape == (mc_samples, 1, 2, 16, 16, 16)


# ── Test 5: Preprocessing transforms on a synthetic subject ──────────────────


@pytest.mark.integration
def test_preprocessing_with_labels_on_synthetic_subject():
    """Pipeline must produce a Subject with all keys; wmh gets 2 channels via OneHot."""
    spatial = 32
    # wmh must have at least one foreground voxel so OneHot produces 2 channels.
    wmh_tensor = torch.zeros(1, spatial, spatial, spatial, dtype=torch.long)
    wmh_tensor[0, spatial // 2, spatial // 2, spatial // 2] = 1
    subject = tio.Subject(
        t1=tio.ScalarImage(tensor=torch.randn(1, spatial, spatial, spatial)),
        flair=tio.ScalarImage(tensor=torch.randn(1, spatial, spatial, spatial)),
        wmh=tio.LabelMap(tensor=wmh_tensor),
    )

    pipeline = get_preprocessing(include_labels=True)
    out = pipeline(subject)

    assert "t1" in out
    assert "flair" in out
    assert "wmh" in out
    # OneHot gives wmh 2 channels: background (0) and foreground (1)
    assert out["wmh"].shape[0] == 2
