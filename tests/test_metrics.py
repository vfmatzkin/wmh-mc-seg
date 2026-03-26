import torch

from src.utils.metrics import compute_metrics

B, S = 2, 8  # batch, spatial (no channel dim needed — function builds one-hot internally)


def _perfect_pred():
    """y_hat == y_true: argmax identical everywhere."""
    # Class 0 wins in first half, class 1 in second half
    y = torch.zeros(B, 2, S, S, S)
    y[:, 0, : S // 2] = 1.0
    y[:, 1, S // 2 :] = 1.0
    return y.clone(), y.clone()


def _random_pred():
    """Purely random softmax predictions."""
    y_hat = torch.softmax(torch.randn(B, 2, S, S, S), dim=1)
    y = torch.softmax(torch.randn(B, 2, S, S, S), dim=1)
    return y_hat, y


def test_compute_metrics_returns_dict_with_dice_key():
    y_hat, y = _perfect_pred()
    result = compute_metrics(y_hat, y)
    assert isinstance(result, dict)
    assert "dice" in result


def test_compute_metrics_respects_prefix():
    y_hat, y = _perfect_pred()
    result = compute_metrics(y_hat, y, text="val_")
    assert "val_dice" in result
    assert "dice" not in result


def test_compute_metrics_perfect_prediction_returns_dice_one():
    y_hat, y = _perfect_pred()
    result = compute_metrics(y_hat, y)
    assert abs(result["dice"] - 1.0) < 1e-4


def test_compute_metrics_random_prediction_returns_dice_below_one():
    torch.manual_seed(0)
    y_hat, y = _random_pred()
    result = compute_metrics(y_hat, y)
    assert result["dice"] < 1.0
