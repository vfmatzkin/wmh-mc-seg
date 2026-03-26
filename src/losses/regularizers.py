import torch
import torch.nn.functional as F


class Regularizers:
    """Regularizers class.

    Penalizes the model for being too confident in erroneous regions.
    Supports MEEP, KL, MEALL, and MEOOD regularization types.

    Note: epsilon is stored as a plain float (device-agnostic). The tensor
    is created on the correct device at call time.
    """

    def __init__(self, epsilon=1e-5, type='MEEP'):
        self.epsilon = epsilon  # plain float, device-agnostic
        self.type = type

    def __call__(self, y_pred, y_true, mask_ood=None, clamp_preds=True):
        y_pred_lm = torch.argmax(y_pred, dim=1)

        eps = torch.tensor(self.epsilon, device=y_pred.device)
        y_pred_c = torch.clamp(y_pred, eps, 1.0 - eps) if clamp_preds else y_pred

        if self.type in ['MEEP', 'KL', 'MEALL']:
            y_true_lm = torch.argmax(y_true, dim=1)
            misclassified_pixels = torch.not_equal(y_pred_lm, y_true_lm).float()
            if self.type == 'MEEP':
                reg = torch.mean(
                    F.binary_cross_entropy(y_pred_c, y_pred_c, reduction="none"),
                    dim=1) * misclassified_pixels
                reg = torch.sum(reg) / torch.sum(misclassified_pixels)
            if self.type == 'MEALL':
                reg = torch.mean(
                    F.binary_cross_entropy(y_pred_c, y_pred_c, reduction="none"),
                    dim=1)
                reg = torch.mean(reg)
            elif self.type == 'KL':
                reg = torch.mean(
                    torch.log(y_pred_c),
                    dim=1) * misclassified_pixels
                reg = torch.sum(reg) / torch.sum(misclassified_pixels)

        if self.type == 'MEOOD':
            # mask_ood marks out-of-distribution images (1 = OOD)
            reg = torch.mean(
                F.binary_cross_entropy(y_pred_c, y_pred_c, reduction="none"),
                dim=1) * mask_ood
            reg = torch.sum(reg) / torch.sum(mask_ood)

        return reg
