import torch
import torch.nn.functional as F
import numpy as np

class Regularizers:
    """ MEEP/KL divergence term for the loss function

    This therm is used as a regularizer for the loss function. It is used to
    penalize the model for being too confident in its predictions in the
    erroneous regions.

    Note that this term has to be

    """

    def __init__(self, epsilon=1e-7, type='MEEP'):
        self.epsilon = torch.tensor(epsilon).cuda() if torch.cuda.is_available() else torch.tensor(epsilon)
        self.type = type

    def __call__(self, y_pred, y_true, mask_ood=None, clamp_preds=True):
        y_pred_lm = torch.argmax(y_pred, dim=1)

        # Clamp the predictions to prevent extreme values
        y_pred_c = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon) \
            if clamp_preds else y_pred

        if self.type in ['MEEP', 'KL']:
            y_true_lm = torch.argmax(y_true, dim=1)
            misclassified_pixels = torch.not_equal(y_pred_lm, y_true_lm).float()
            if self.type == 'MEEP':
                reg = torch.mean(
                    F.binary_cross_entropy(y_pred_c, y_pred_c,
                                           reduction="none"),
                    dim=1) * misclassified_pixels
                reg = torch.sum(reg) / torch.sum(misclassified_pixels)
            elif self.type == 'KL':
                reg = torch.mean(
                    torch.log(y_pred_c),
                    dim=1) * misclassified_pixels
                reg = torch.sum(reg) / torch.sum(misclassified_pixels)

        if self.type == 'MEOOD':
            # mask_ood is a list of 0s and 1s, where 1s indicate that the
            # corresponding image is out of distribution
            reg = torch.mean(
                F.binary_cross_entropy(y_pred_c, y_pred_c,
                                        reduction="none"),
                dim=1) * mask_ood
            reg = torch.sum(reg) / torch.sum(mask_ood)
        return reg


class BCEMEEPLoss(torch.nn.Module):
    def __init__(self, start_on_epoch=0, reg_lambda=0.3):
        """ Cross Entropy + MEEP Loss

        This loss function is a combination of the Binary Cross Entropy and the
        Maximum Entropy on Erroneous Predictions (MEEP) loss term.

        In the paper, meep_lambda is set to 0.3 for Cross Entropy.
        https://arxiv.org/pdf/2112.12218.pdf
        """
        super().__init__()
        self.CE = torch.nn.CrossEntropyLoss()
        self.MEEP = Regularizers(type='MEEP')
        self.m_lambda = reg_lambda
        self.start_on_epoch = start_on_epoch

    def forward(self, y_pred, y_true, epoch, **kwargs):
        use_meep = epoch >= self.start_on_epoch

        ce = self.CE(y_pred, y_true)
        meep = self.MEEP(y_pred, y_true) if use_meep else 0

        return {'ce': ce, 'meep': -self.m_lambda * meep}


class BCEKLLoss(torch.nn.Module):
    def __init__(self, start_on_epoch=0, reg_lambda=0.3):
        """ Cross Entropy + KL Loss

        This loss function is a combination of the Binary Cross Entropy and the
        Maximum Entropy on Erroneous Predictions (MEEP) loss term.

        In the paper, meep_lambda is set to 0.3 for Cross Entropy.
        https://arxiv.org/pdf/2112.12218.pdf
        """
        super().__init__()
        self.CE = torch.nn.CrossEntropyLoss()
        self.KL = Regularizers(type='KL')
        self.m_lambda = reg_lambda
        self.start_on_epoch = start_on_epoch

    def forward(self, y_pred, y_true, epoch, **kwargs):
        use_kl = epoch >= self.start_on_epoch

        ce = self.CE(y_pred, y_true)
        kl = self.KL(y_pred, y_true) if use_kl else 0

        return {'ce': ce, 'kl': -self.m_lambda * kl}


class CEMEOODLoss(torch.nn.Module):
    """ Cross Entropy + MEOOD Loss
    This loss computes the Cross Entropy loss if the image is in-distribution,
    and applies maximum entropy on out of distribution images.
    """
    def __init__(self, start_on_epoch=0, reg_lambda=1, ood_centers=None):
        super().__init__()
        self.CE = torch.nn.CrossEntropyLoss()
        self.MEOOD = Regularizers(type='MEOOD')
        self.m_lambda = reg_lambda
        self.start_on_epoch = start_on_epoch
        self.ood_centers = ood_centers.split(',') if ood_centers else None

    def forward(self, y_pred, y_true, epoch, centers, **kwargs):
        use_reg = epoch >= self.start_on_epoch

        centers_array = np.array(centers)
        batch_size, channels, height, width, depth = y_pred.shape
        mask = torch.tensor([[[[1 if centers_array[i] in self.ood_centers else 0
                                for _ in range(width)] for _ in range(height)]
                              for _ in range(depth)] for i in
                             range(batch_size)], device=y_pred.device)
        ce = self.CE(y_pred, y_true)
        meood = self.MEOOD(y_pred, y_true, mask) if use_reg else 0

        return {'ce': ce, 'meood': -self.m_lambda * meood}