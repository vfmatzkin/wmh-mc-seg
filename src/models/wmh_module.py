import csv
import os
import shutil

import mlflow
import lightning as L
import torch
import torch.nn.functional as F
import torchio as tio

from losses.composite import RegularizedLoss
from models.inference import (
    forward_pass,
    get_mc_preds,
    save_predictions,
)
from models.unet3d import UNet3D
from utils.metrics import compute_metrics


class WMHModel(L.LightningModule):
    """ WMHModel

    This class implements the WMH segmentation model. It is a subclass of the
    LightningModule class from PyTorch Lightning. This class is used to train
    the model and to save the predictions to the disk.

    :param net: Model instance
    :param criterion: Loss function name (resolved via RegularizedLoss.from_cli).
    :param learning_rate: Learning rate
    :param optimizer_class: Optimizer class
    :param weight_decay: Weight decay
    :param lambda_lr: LambdaLR coefficient (scheduler)
    :param reduce_on_epoch: Reduce the learning rate every N epochs.
    """

    def __init__(self, net, criterion, learning_rate, optimizer_class,
                 weight_decay=0, lambda_lr=None, reduce_on_epoch=None,
                 reg_start=0, reg_lambda=0.3, best_model_path=None,
                 ood_centers=None, **kwargs):
        super().__init__()

        self.lr = learning_rate
        self.net = net
        self.criterion, self.custom_loss = RegularizedLoss.from_cli(
            criterion, reg_lambda=reg_lambda, start_epoch=reg_start,
            ood_centers=ood_centers,
        )
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay
        self.lambda_lr = lambda_lr
        self.reduce_on_epoch = reduce_on_epoch
        if best_model_path is not None:
            self.best_model_path = os.path.abspath(best_model_path)
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        self.best_model_path = best_model_path

        # Test-related parameters
        self.model_path = None
        self.save_preds = kwargs.get('save_predictions', False)
        self.output_dir = kwargs.get('output_dir', None)
        self.saved_preds = []
        self.patch_size = kwargs.get('patch_size', None)
        self.mc_dropout_ratio = kwargs.get('mc_dropout_ratio', 0.0)
        self.mc_dropout_samples = kwargs.get('mc_dropout_samples', 0)

        self.save_hyperparameters(ignore=['net'])

    @classmethod
    def load_test(cls, model_path, save_predictions, output_dir, patch_size,
                  mc_dropout_ratio, mc_dropout_samples):
        model_path = os.path.expanduser(model_path)
        obj = WMHModel.load_from_checkpoint(model_path,
                                            net=UNet3D(mc_dropout_ratio))
        obj.model_path = os.path.abspath(model_path)
        obj.save_preds = save_predictions
        obj.output_dir = output_dir
        obj.patch_size = patch_size
        obj.mc_dropout_ratio = mc_dropout_ratio
        obj.mc_dropout_samples = mc_dropout_samples

        return obj

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)
        if self.lambda_lr is not None and self.reduce_on_epoch is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: self.lambda_lr ** (
                        epoch // self.reduce_on_epoch),
            )
            return [optimizer], [scheduler]
        return optimizer

    def forward_pass(self, x, batch, is_test=False):
        return forward_pass(self.net, x, batch, is_test, self.patch_size)

    def infer_batch(self, batch, is_test=False):
        xc1 = batch['t1'][tio.DATA]
        xc2 = batch['flair'][tio.DATA]
        t1_paths = batch['t1'][tio.PATH]
        y = batch['wmh'][tio.DATA] if 'wmh' in batch else None

        base_folder = os.path.commonprefix(t1_paths)
        centers = []

        # If all the images are from the same center, remove the center from the
        # base folder
        if not any(spl in os.path.split(base_folder.rstrip('/'))[1]
                   for spl in ['training', 'test']):
            base_folder = os.path.split(base_folder.rstrip('/'))[0]

        for file_path in t1_paths:
            relative_path = file_path.replace(base_folder, '').lstrip('/')
            path_parts = relative_path.split('/')
            center_part = path_parts[0]
            centers.append(center_part)

        # Concatenate the input images along the channel dimension
        x = torch.cat((xc1, xc2), dim=1)

        # Get MC dropout predictions if it applies (logits and softmax)
        lgs_mc_arr, sm_mc_arr = get_mc_preds(
            self.net, x, batch, self.mc_dropout_ratio, self.mc_dropout_samples,
            self.patch_size, is_test,
        )

        # Forward pass
        logits = self.forward_pass(x, batch, is_test)
        y_hat = F.softmax(logits, dim=1)

        # Save predictions if it applies
        saved = save_predictions(
            y_hat, y, logits, t1_paths,
            model_path=self.model_path,
            output_dir=self.output_dir,
            save_preds=self.save_preds,
            mc_dropout_samples=self.mc_dropout_samples,
            is_test=is_test,
            lgs_mc_arr=lgs_mc_arr,
            sm_mc_arr=sm_mc_arr,
        )
        if saved:
            self.saved_preds.extend(saved)

        return y_hat, y, centers  # Return preds, gt and centers

    def _log_losses(self, losses, stage):
        epoch = self.current_epoch
        loss = 0
        if isinstance(losses, dict):
            for key in losses:
                self.log(f'{stage}_{key}', losses[key], prog_bar=True,
                         on_step=True, on_epoch=True)
                mlflow.log_metric(f'{stage}_{key}', losses[key], epoch)
                loss += losses[key]
        else:
            loss = losses
        return loss

    def _shared_step(self, batch, batch_idx, stage):
        y_hat, y, centers = self.infer_batch(batch)
        losses = self.criterion(y_hat, y, self.current_epoch, centers=centers) \
            if self.custom_loss else self.criterion(y_hat, y)
        metrics = compute_metrics(y_hat, y, text=f'{stage}_')
        loss = self._log_losses(losses, stage)
        self.log(f'{stage}_loss', loss, prog_bar=True, on_step=True,
                 on_epoch=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.infer_batch(batch, is_test=True)

    def on_train_epoch_end(self):  # Todo change filtering _epoch
        metr, ce = self.trainer.callback_metrics, self.current_epoch
        if 'train_loss_epoch' in metr:
            mlflow.log_metric('train_loss', metr['train_loss_epoch'].item(), ce)
        if 'train_dice_epoch' in metr:
            mlflow.log_metric('train_dice', metr['train_dice_epoch'].item(), ce)

    def on_validation_epoch_end(self):
        metr, ce = self.trainer.callback_metrics, self.current_epoch
        if 'val_loss_epoch' in metr:
            mlflow.log_metric('val_loss', metr['val_loss_epoch'].item(), ce)
        if 'val_dice_epoch' in metr:
            mlflow.log_metric('val_dice', metr['val_dice_epoch'].item(), ce)
        best_chk_path = self.trainer.checkpoint_callback.best_model_path
        if best_chk_path != '' and self.best_model_path is not None:
            shutil.copy(best_chk_path, self.best_model_path)

    def save_preds_info(self, preds_info_path, model_path, force_save=False):
        """ Saves the paths to the saved predictions

        :param preds_info_path: Path to the file where the paths will be saved
        :param model_path: Path to the model checkpoint
        :param force_save: Whether to save the predictions even if the
        preds_info_path is None
        """
        if preds_info_path is None and force_save:
            folder_name = '_'.join(
                os.path.basename(model_path).split('_')[0:-1])
            file_name = os.path.basename(model_path).replace('.ckpt', '.csv')
            preds_info_path = os.path.join(os.getcwd(), 'notebooks',
                                           folder_name, file_name)
        elif preds_info_path is None:  # Not saving predictions csv by default
            return None

        preds_info_path = os.path.abspath(os.path.expanduser(preds_info_path))
        os.makedirs(os.path.dirname(preds_info_path), exist_ok=True)

        with open(preds_info_path, 'w', ) as f:
            writer = csv.writer(f)
            writer.writerows(self.saved_preds)
        print(f'Predictions info saved to {preds_info_path}')
