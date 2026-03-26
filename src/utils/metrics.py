from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.metrics import compute_dice as dice


def compute_metrics(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    text: str = "",
) -> dict[str, float]:
    """Computes the metrics

    This function computes the metrics that are used to evaluate the model
    while training.

    :param y_hat: Predicted labels
    :param y: Ground truth labels
    :param text: Text to prepend to the metric name
    :return:
    """
    # Prepare one-hot encoded tensors for the dice score computation
    y_hat_one_hot = F.one_hot(torch.argmax(y_hat, dim=1), num_classes=2)
    y_one_hot = F.one_hot(torch.argmax(y, dim=1), num_classes=2)
    y_hat_perm = torch.permute(y_hat_one_hot, [0, 4, 1, 2, 3])
    y_perm = torch.permute(y_one_hot, [0, 4, 1, 2, 3])

    # Compute the dice score
    dice_score = torch.mean(dice(y_hat_perm, y_perm, ignore_empty=False))

    # Construct the result dictionary
    result = {text + "dice": float(dice_score)}

    return result
