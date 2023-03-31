import sys
import tempfile

import mlflow
import monai
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchio as tio


def prevent_logging(force=False, debug=True):
    """ Prevents MLFlow from logging if debugging or force flag is set

    :param force: Force disable logging
    :param debug: If True, logging will be disabled if debugging is detected
    (default: True)
    """
    get_trace = getattr(sys, 'gettrace', None)
    if (get_trace is not None and get_trace() and not debug) or force:
        temp_dir = tempfile.TemporaryDirectory()
        print("Debugging detected or force flag set. MLFlow won't log.")
        mlflow.set_tracking_uri(temp_dir.name)
    else:
        print("MLFlow will log to the default location")


def prepare_batch(batch):
    return batch['t1'][tio.DATA], batch['flair'][tio.DATA], \
        batch['wmh'][tio.DATA]


def compute_metrics(y_hat, y, text=''):
    met = {
        text + 'dice': torch.mean(monai.metrics.compute_dice(
            torch.permute(
                F.one_hot(torch.argmax(y_hat, dim=1),
                          num_classes=2),
                [0, 4, 1, 2, 3]
            ),
            torch.permute(
                F.one_hot(torch.argmax(y, dim=1),
                          num_classes=2),
                [0, 4, 1, 2, 3]
            )
            , ignore_empty=False)),  # TODO Check what to do with empty patches
    }
    return met


class WMHModel(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        """ WMHModel

        :param net: Model instance
        :param criterion: Loss function
        :param learning_rate: Learning rate
        :param optimizer_class: Optimizer class
        """
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

        self.save_hyperparameters(ignore=['net', 'criterion'])

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def infer_batch(self, batch):
        xc1, xc2, y = prepare_batch(batch)
        x = torch.cat((xc1, xc2), dim=1)  # Concatenate the channels
        y_hat = F.softmax(self.net(x), dim=1)  # TODO Softmax here or in loss?
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        metrics = compute_metrics(y_hat, y, text='train_')

        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        metrics = compute_metrics(y_hat, y, text='val_')

        self.log('val_loss', loss)
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)

        if y is not None:
            loss = self.criterion(y_hat, y)
            metrics = compute_metrics(y_hat, y, text='test_')

            self.log('test_loss', loss)
            self.log_dict(metrics)
            return loss
        return None