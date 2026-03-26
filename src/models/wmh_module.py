import csv
import os
import shutil

import SimpleITK as sitk
import mlflow
import lightning as L
import torch
import torch.nn.functional as F
import torchio as tio

from losses.composite import RegularizedLoss
from models.unet3d import UNet3D
from utils.metrics import compute_metrics
from utils.sitk_io import restore_metadata_as_sitk


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

    def get_pred_folder(self, t1_path):
        parent_folder = os.path.dirname(os.path.dirname(t1_path))
        if self.output_dir is None:
            pred_folder = parent_folder
        else:
            pred_folder = os.path.join(self.output_dir,
                                       os.path.basename(parent_folder))
        os.makedirs(pred_folder, exist_ok=True)
        return pred_folder

    def save_preds_tst(self, y_hat, y, logits, reference_imgs, is_test=False,
                       lgs_mc_arr=None, sm_mc_arr=None):
        """ Saves predictions to the disk

        It expects full volume images, not patches.

        :param y_hat: Softmax output
        :param y: Ground truth
        :param logits: Logits
        :param reference_imgs: Paths to the T1 images
        :param is_test: Whether the model is being tested or not
        :param lgs_mc_arr: Logits array (for MC dropout)
        :param sm_mc_arr: Softmax output array (for MC dropout)
        """
        if not is_test or (not self.save_preds and self.output_dir is None):
            return None

        for i, t1_path in enumerate(reference_imgs):  # For each img in batch
            pred_folder = self.get_pred_folder(t1_path)
            hard_pred = torch.argmax(y_hat[i], dim=0).to(torch.uint8)

            lgs_mc_mean = lgs_mc_arr.mean(0) if lgs_mc_arr is not None else None
            sftmx_mc_mean = sm_mc_arr.mean(0) if sm_mc_arr is not None else None
            hard_pred_mc, uncert_mc = None, None
            if sm_mc_arr is not None:
                hard_pred_mc = torch.argmax(sftmx_mc_mean[0], 0).to(torch.uint8)
                if self.mc_dropout_samples == 1:
                    print('WARNING: MC dropout samples == 1, uncertainty '
                          'estimation won\'t be computed')
                uncert_mc = sm_mc_arr.std(0)[0] if self.mc_dropout_samples > 1 \
                    else None  # First elem in batch + foreground ch on save

            run_id = os.path.splitext(os.path.basename(self.model_path))[0]
            imgs = {
                f'pred_wmh_hard_{run_id}.nii.gz': hard_pred,
                f'pred_wmh_softmax_{run_id}.nii.gz': y_hat[i],
                f'pred_logits_{run_id}.nii.gz': logits[i],
                f'gt_wmh_{run_id}.nii.gz':
                    y[i, 1].to(torch.uint8) if y is not None else None,
            }

            if lgs_mc_arr.shape[0]:  # If there's any MC imgs to save
                imgs.update({
                    f'pred_mc_logitsmean_{run_id}.nii.gz': lgs_mc_mean[0],
                    f'pred_mc_softmaxmean_{run_id}.nii.gz': sftmx_mc_mean[0],
                    f'pred_mc_hardmean_{run_id}.nii.gz': hard_pred_mc,
                })

            if uncert_mc is not None:  # If there's any MC imgs to save and > 1
                imgs.update({
                    f'pred_mc_uncertmc_{run_id}.nii.gz': uncert_mc[1]
                })

            paths = []
            for img_name, img in imgs.items():
                paths.append(os.path.join(pred_folder, img_name))
                if img is not None:
                    im_o_sp = restore_metadata_as_sitk(img, t1_path)
                    sitk.WriteImage(im_o_sp, paths[-1])
            self.saved_preds.append(paths)
            print(f'saved preds for {pred_folder}')

    def forward_pass(self, x, batch, is_test=False):
        if not is_test or self.patch_size is None:
            return self.net(x)
        else:
            return self.patch_inference(x, batch)

    def patch_inference(self, x, batch):
        out_tensor = torch.empty((0, x.shape[1], x.shape[2], x.shape[3],
                                  x.shape[4]))
        for i_subj in range(x.shape[0]):
            if 'wmh' in batch:
                subject = tio.Subject(
                    t1=tio.ScalarImage(tensor=batch['t1']['data'][i_subj]),
                    flair=tio.ScalarImage(
                        tensor=batch['flair']['data'][i_subj]
                    ),
                    wmh=tio.LabelMap(tensor=batch['wmh']['data'][i_subj]))
            grid_sampler = tio.inference.GridSampler(
                subject,
                self.patch_size,
                4,
            )
            patch_loader = torch.utils.data.DataLoader(
                grid_sampler,
                batch_size=1,
            )
            aggregator = tio.inference.GridAggregator(grid_sampler)
            with torch.no_grad():
                for patches_batch in patch_loader:
                    xc1 = patches_batch['t1'][tio.DATA]
                    xc2 = patches_batch['flair'][tio.DATA]

                    # Concatenate the input images along the channel dimension
                    x = torch.cat((xc1, xc2), dim=1)

                    locations = patches_batch[tio.LOCATION]

                    logits = self.net(x)
                    aggregator.add_batch(logits, locations)
            output_tensor = aggregator.get_output_tensor()
            # Append it to the out_tensor in the first channel (batch)
            # out_tensor is a 5D tensor (batch, channels, x, y, z)
            # and output_tensor is a 4D tensor (channels, x, y, z)
            out_tensor = torch.cat((out_tensor, output_tensor.unsqueeze(0)))
        return out_tensor

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
        lgs_mc_arr, sm_mc_arr = self.get_mc_preds(x, batch, is_test)

        # Forward pass
        logits = self.forward_pass(x, batch, is_test)
        y_hat = F.softmax(logits, dim=1)

        # Save predictions if it applies
        self.save_preds_tst(y_hat, y, logits, t1_paths, is_test, lgs_mc_arr,
                            sm_mc_arr)
        return y_hat, y, centers  # Return preds, gt and centers

    def get_mc_preds(self, x, batch, is_test):
        # For MC dropout:
        mc_cond = self.mc_dropout_ratio > 0 and is_test
        if mc_cond:
            model_was_eval = not self.net.training
            self.net.train()

        # If no MC dropout -> return empty arrays
        mc_fwd_passes = self.mc_dropout_samples if mc_cond else 0
        logits_arr = torch.empty((mc_fwd_passes, x.shape[0], 2, x.shape[2],
                                  x.shape[3], x.shape[4]))
        y_hat_arr = torch.empty((mc_fwd_passes, x.shape[0], 2, x.shape[2],
                                 x.shape[3], x.shape[4]))

        # MC forward passes
        for i in range(mc_fwd_passes):
            print(f'  - MC dropout: forward pass {i + 1}/{mc_fwd_passes}')
            logits_mc = self.forward_pass(x, batch, is_test)
            y_hat_mc = F.softmax(logits_mc, dim=1)
            logits_arr[i] = logits_mc
            y_hat_arr[i] = y_hat_mc

        # Restore the model to its original state
        if mc_cond and model_was_eval:
            self.net.eval()

        return logits_arr, y_hat_arr

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
