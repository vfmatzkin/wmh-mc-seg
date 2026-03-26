import pytest
import torch

from src.losses.composite import RegularizedLoss, CLI_ALIASES
from src.losses.regularizers import Regularizers


# ── helpers ──────────────────────────────────────────────────────────────────

B, C, S = 2, 2, 4  # batch, channels, spatial


def make_tensors():
    """Small random tensors: (B, C, S, S, S), values in [0, 1]."""
    y_pred = torch.softmax(torch.randn(B, C, S, S, S), dim=1)
    y_true = torch.softmax(torch.randn(B, C, S, S, S), dim=1)
    return y_pred, y_true


# ── from_cli: alias resolution ────────────────────────────────────────────────

@pytest.mark.parametrize('alias', list(CLI_ALIASES.keys()))
def test_from_cli_all_aliases_resolve(alias):
    loss, _ = RegularizedLoss.from_cli(alias)
    assert isinstance(loss, RegularizedLoss)


def test_from_cli_non_regularized_aliases_is_custom_false():
    plain_aliases = [k for k, (_, reg) in CLI_ALIASES.items() if reg is None]
    for alias in plain_aliases:
        _, is_custom = RegularizedLoss.from_cli(alias)
        assert not is_custom, f"Expected is_custom=False for '{alias}'"


def test_from_cli_regularized_aliases_is_custom_true():
    reg_aliases = [k for k, (_, reg) in CLI_ALIASES.items() if reg is not None]
    for alias in reg_aliases:
        _, is_custom = RegularizedLoss.from_cli(alias)
        assert is_custom, f"Expected is_custom=True for '{alias}'"


def test_from_cli_ce_alias_maps_to_ce():
    loss, is_custom = RegularizedLoss.from_cli('ce')
    assert not is_custom
    assert loss.regularizer is None


def test_from_cli_meep_alias_maps_to_ce_plus_meep():
    loss, is_custom = RegularizedLoss.from_cli('meep')
    assert is_custom
    assert loss.regularizer is not None
    assert loss.regularizer.type == 'MEEP'


def test_from_cli_kl_alias_maps_to_ce_plus_kl():
    loss, is_custom = RegularizedLoss.from_cli('kl')
    assert is_custom
    assert loss.regularizer.type == 'KL'


def test_from_cli_meall_alias_maps_to_ce_plus_meall():
    loss, is_custom = RegularizedLoss.from_cli('meall')
    assert is_custom
    assert loss.regularizer.type == 'MEALL'


def test_from_cli_dicemeep_alias():
    loss, is_custom = RegularizedLoss.from_cli('dicemeep')
    assert is_custom
    assert loss.regularizer.type == 'MEEP'


def test_from_cli_raises_for_unknown_name():
    with pytest.raises(ValueError, match='Unknown loss function'):
        RegularizedLoss.from_cli('not_a_real_loss')


# ── forward: return type ──────────────────────────────────────────────────────

def test_forward_non_regularized_returns_tensor():
    loss, _ = RegularizedLoss.from_cli('ce')
    y_pred, y_true = make_tensors()
    # CrossEntropyLoss expects class indices, not one-hot
    y_true_idx = torch.argmax(y_true, dim=1)
    result = loss(y_pred, y_true_idx, epoch=0)
    assert isinstance(result, torch.Tensor)


def test_forward_dice_non_regularized_returns_tensor():
    loss, _ = RegularizedLoss.from_cli('dice')
    y_pred, y_true = make_tensors()
    result = loss(y_pred, y_true, epoch=0)
    assert isinstance(result, torch.Tensor)


def test_forward_regularized_returns_dict():
    loss, _ = RegularizedLoss.from_cli('meep')
    y_pred, y_true = make_tensors()
    result = loss(y_pred, y_true, epoch=0)
    assert isinstance(result, dict)
    assert 'base' in result
    assert 'reg' in result


def test_forward_regularized_before_start_epoch_returns_dict_without_reg():
    loss, _ = RegularizedLoss.from_cli('meep')
    loss.start_epoch = 100  # push start far out
    y_pred, y_true = make_tensors()
    result = loss(y_pred, y_true, epoch=0)
    assert isinstance(result, dict)
    assert 'base' in result
    assert 'reg' not in result


# ── Regularizers: device placement ───────────────────────────────────────────

def test_regularizers_epsilon_created_on_cpu():
    reg = Regularizers(type='MEEP')
    y_pred = torch.softmax(torch.randn(B, C, S, S, S), dim=1)
    y_true = torch.softmax(torch.randn(B, C, S, S, S), dim=1)
    # Should not raise — epsilon tensor must be created on y_pred.device (cpu)
    result = reg(y_pred, y_true)
    assert isinstance(result, torch.Tensor)
    assert result.device.type == 'cpu'
