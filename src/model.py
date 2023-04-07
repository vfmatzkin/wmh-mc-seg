import csv
import os

import SimpleITK as sitk
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchio as tio
from monai.metrics import compute_dice as dice


def compute_metrics(y_hat, y, text=''):
    """ Computes the metrics

    This function computes the metrics that are used to evaluate the model while
    training.

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
    result = {text + 'dice': dice_score}

    return result


class WMHModel(pl.LightningModule):
    """ WMHModel

    This class implements the WMH segmentation model. It is a subclass of the
    LightningModule class from PyTorch Lightning. This class is used to train
    the model and to save the predictions to the disk.

    :param net: Model instance
    :param criterion: Loss function(s)
    :param learning_rate: Learning rate
    :param optimizer_class: Optimizer class
    """

    def __init__(self, net, criterion, learning_rate, optimizer_class,
                 **kwargs):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

        # Test-related parameters
        self.run_id = kwargs.get('run_id', None)
        self.save_preds = kwargs.get('save_predictions', False)
        self.output_dir = kwargs.get('output_dir', None)
        self.saved_preds = []

        # TODO Remove when removing autolog
        self.save_hyperparameters(ignore=['net', 'criterion'])

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def get_pred_folder(self, t1_path):
        parent_folder = os.path.dirname(os.path.dirname(t1_path))
        if self.output_dir is None:
            pred_folder = parent_folder
        else:
            pred_folder = os.path.join(self.output_dir,
                                       os.path.basename(parent_folder))
        os.makedirs(pred_folder, exist_ok=True)
        return pred_folder

    def save_predictions(self, y_hat, y, logits, t1_paths):
        """ Saves predictions to the disk

        It expects full volume images, not patches.

        :param y_hat: Softmax output
        :param y: Ground truth
        :param logits: Logits
        :param t1_paths: Paths to the T1 images
        """
        if not self.save_preds and self.output_dir is None:
            return None

        for i, t1_path in enumerate(t1_paths):  # For each image in the batch
            pred_folder = self.get_pred_folder(t1_path)
            hard_pred = torch.argmax(y_hat[i], dim=0).to(torch.uint8)

            imgs = {
                f'pred_wmh_hard_{self.run_id}.nii.gz': hard_pred,
                f'pred_wmh_softmax_{self.run_id}.nii.gz': y_hat[i, 1],
                f'pred_logits_{self.run_id}.nii.gz': logits[i, 1],
                f'gt_wmh_{self.run_id}.nii.gz':
                    y[i, 1].to(torch.uint8) if y is not None else None,
            }

            paths = []
            for img_name, img in imgs.items():
                paths.append(os.path.join(pred_folder, img_name))
                if img is not None:
                    sitk.WriteImage(sitk.GetImageFromArray(img),
                                    paths[-1])
            self.saved_preds.append(paths)
            print(f'saved preds for {pred_folder}')

    def infer_batch(self, batch, save_preds=False):
        xc1 = batch['t1'][tio.DATA]
        xc2 = batch['flair'][tio.DATA]
        y = batch['wmh'][tio.DATA] if 'wmh' in batch else None

        # Concatenate the input images along the channel dimension
        x = torch.cat((xc1, xc2), dim=1)

        # Forward pass through the neural network
        logits = self.net(x)
        y_hat = F.softmax(logits, dim=1)

        if save_preds:  # Save the predictions to the disk (for testing)
            t1_paths = batch['t1'][tio.PATH]
            self.save_predictions(y_hat, y, logits, t1_paths)

        return y_hat, y  # Return the predicted and GT masks

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
        y_hat, y = self.infer_batch(batch, save_preds=True)

    def save_preds_info(self, preds_info_path):
        """ Saves the paths to the saved predictions

        :param preds_info_path: Path to the file where the paths will be saved
        """
        with open(preds_info_path, 'w', ) as f:
            writer = csv.writer(f)
            writer.writerows(self.saved_preds)
