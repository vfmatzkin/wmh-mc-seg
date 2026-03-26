from __future__ import annotations

import numpy as np
import torch
from monai.losses import DiceCELoss, DiceLoss, FocalLoss

from .regularizers import Regularizers

LOSS_REGISTRY = {
    "ce": lambda: torch.nn.CrossEntropyLoss(),
    "dice": lambda: DiceLoss(),
    "dicece": lambda: DiceCELoss(),
    "focal": lambda: FocalLoss(),
}

REG_REGISTRY = {
    "meep": "MEEP",
    "kl": "KL",
    "meall": "MEALL",
    "meood": "MEOOD",
}

# Maps CLI --loss name -> (base_key, reg_key)
CLI_ALIASES = {
    "crossentropy": ("ce", None),
    "ce": ("ce", None),
    "dice": ("dice", None),
    "dicece": ("dicece", None),
    "focal": ("focal", None),
    "meep": ("ce", "meep"),
    "cemeep": ("ce", "meep"),
    "crosentropymeep": ("ce", "meep"),
    "dicemeep": ("dice", "meep"),
    "dmeep": ("dice", "meep"),
    "cekl": ("ce", "kl"),
    "kl": ("ce", "kl"),
    "crosentropykl": ("ce", "kl"),
    "dicekl": ("dice", "kl"),
    "cemeall": ("ce", "meall"),
    "meall": ("ce", "meall"),
    "crosentropymeall": ("ce", "meall"),
    "dicemeall": ("dice", "meall"),
    "meood": ("ce", "meood"),
    "cemeood": ("ce", "meood"),
}


class RegularizedLoss(torch.nn.Module):
    """Unified loss wrapper combining a base loss with an optional regularizer.

    For non-regularized losses, forward() returns a bare Tensor.
    For regularized losses, forward() returns a dict[str, Tensor] with 'base'
    and 'reg' keys (reg is only present once epoch >= start_epoch).

    The MEOOD regularizer requires a 'centers' kwarg in forward() and uses
    ood_centers to build the per-image OOD mask.
    """

    def __init__(
        self,
        base_loss: torch.nn.Module,
        regularizer: Regularizers | None = None,
        reg_lambda: float = 0.3,
        start_epoch: int = 0,
        ood_centers: list[str] | str | None = None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.regularizer = regularizer
        self.reg_lambda = reg_lambda
        self.start_epoch = start_epoch
        # ood_centers may be a comma-separated string or a list
        if isinstance(ood_centers, str):
            self.ood_centers = ood_centers.split(",")
        else:
            self.ood_centers = ood_centers  # list or None

    @classmethod
    def from_cli(
        cls,
        loss_name: str,
        reg_lambda: float = 0.3,
        start_epoch: int = 0,
        ood_centers: list[str] | str | None = None,
    ) -> tuple[RegularizedLoss, bool]:
        """Build a RegularizedLoss from a CLI loss name.

        Returns (loss_instance, is_custom) where is_custom is True when a
        regularizer is attached.

        Raises ValueError for unknown loss names.
        """
        key = loss_name.lower()
        if key not in CLI_ALIASES:
            raise ValueError(f"Unknown loss function: {loss_name}")
        base_key, reg_key = CLI_ALIASES[key]
        base = LOSS_REGISTRY[base_key]()
        reg = Regularizers(type=REG_REGISTRY[reg_key]) if reg_key else None
        is_custom = reg_key is not None
        return cls(base, reg, reg_lambda, start_epoch, ood_centers), is_custom

    def _build_ood_mask(self, y_pred: torch.Tensor, centers: list[str]) -> torch.Tensor:
        """Build per-voxel OOD mask from batch centers list."""
        centers_array = np.array(centers)
        batch_size, _channels, height, width, depth = y_pred.shape
        mask = torch.tensor(
            [
                [
                    [
                        [1 if centers_array[i] in self.ood_centers else 0 for _ in range(width)]
                        for _ in range(height)
                    ]
                    for _ in range(depth)
                ]
                for i in range(batch_size)
            ],
            device=y_pred.device,
        )
        return mask

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        epoch: int,
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.regularizer is None:
            return self.base_loss(y_pred, y_true)

        # Regularized path
        if self.regularizer.type == "MEOOD":
            centers = kwargs.get("centers")
            mask_ood = self._build_ood_mask(y_pred, centers)
            inv_msk = 1 - mask_ood
            base_val = self.base_loss(
                y_pred * inv_msk.unsqueeze(1),
                y_true * inv_msk.unsqueeze(1),
            )
        else:
            mask_ood = kwargs.get("mask_ood")
            base_val = self.base_loss(y_pred, y_true)

        result = {"base": base_val}
        if epoch >= self.start_epoch:
            reg_val = self.regularizer(y_pred, y_true, mask_ood)
            result["reg"] = -self.reg_lambda * reg_val

        return result
