import torch
import torch.nn.functional as F


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

    def __call__(self, y_pred, y_true):
        y_pred_lm = torch.argmax(y_pred, dim=1)
        y_true_lm = torch.argmax(y_true, dim=1)
        misclassified_pixels = torch.not_equal(y_pred_lm, y_true_lm).float()

        # Clamp the predictions to prevent extreme values
        y_pred_clamped = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)

        if self.type == 'MEEP':
            reg = torch.mean(
                F.binary_cross_entropy(y_pred_clamped, y_pred_clamped, reduction="none"),
                dim=1) * misclassified_pixels
            reg = torch.sum(reg) / torch.sum(misclassified_pixels)
        elif self.type == 'KL':
            reg = torch.mean(
                torch.log(y_pred_clamped),
                dim=1) * misclassified_pixels
            reg = torch.sum(reg) / torch.sum(misclassified_pixels)
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

    def forward(self, y_pred, y_true, epoch):
        use_meep = epoch >= self.start_on_epoch

        # print(f'Using MEEP: {use_meep}'
        #       f' (epoch: {epoch}, start_on_epoch: {self.start_on_epoch})'
        #       f' (m_lambda: {self.m_lambda})')

        ce = self.CE(y_pred, y_true)
        meep = self.MEEP(y_pred, y_true) if use_meep else 0

        # print(f'CE: {ce}, MEEP: {meep}')

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

    def forward(self, y_pred, y_true, epoch):
        use_kl = epoch >= self.start_on_epoch

        ce = self.CE(y_pred, y_true)
        kl = self.KL(y_pred, y_true) if use_kl else 0

        return {'ce': ce, 'kl': -self.m_lambda * kl}
