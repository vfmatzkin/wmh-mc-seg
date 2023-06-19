import torch
import torch.nn.functional as F


class MEEP:
    def __call__(self, y_pred, y_true):
        y_pred_lm = torch.argmax(y_pred, dim=1)
        y_true_lm = torch.argmax(y_true, dim=1)
        misclassified_pixels = torch.not_equal(y_pred_lm, y_true_lm).float()
        entropy = torch.mean(
            F.binary_cross_entropy(y_pred, y_true, reduction="none"),
            dim=1) * misclassified_pixels
        entropy = torch.sum(entropy) / torch.sum(misclassified_pixels)
        return entropy


class BCEMEEPLoss(torch.nn.Module):
    def __init__(self, start_on_epoch=0, meep_lambda=0.3):
        """ Cross Entropy + MEEP Loss

        This loss function is a combination of the Binary Cross Entropy and the
        Maximum Entropy on Erroneous Predictions (MEEP) loss term.

        In the paper, meep_lambda is set to 0.3 for Cross Entropy.
        https://arxiv.org/pdf/2112.12218.pdf
        """
        super().__init__()
        self.CE = torch.nn.CrossEntropyLoss()
        self.MEEP = MEEP()
        self.m_lambda = meep_lambda
        self.start_on_epoch = start_on_epoch

    def forward(self, y_pred, y_true, epoch):
        use_meep = epoch >= self.start_on_epoch

        print(f'Using MEEP: {use_meep}'
              f' (epoch: {epoch}, start_on_epoch: {self.start_on_epoch})'
              f' (m_lambda: {self.m_lambda})')

        ce = self.CE(y_pred, y_true)
        meep = self.MEEP(y_pred, y_true) if use_meep else 0

        print(f'CE: {ce}, MEEP: {meep}')

        return ce + self.m_lambda * meep
