import csv
import os

import SimpleITK as sitk
import mlflow
import monai
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import compute_dice as dice
from torch.nn import CrossEntropyLoss

from losses import BCEMEEPLoss


def compute_metrics(y_hat, y, text=''):
    """ Computes the metrics

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
    result = {text + 'dice': float(dice_score)}

    return result


class UNet3D(monai.networks.nets.UNet):
    def __init__(self, dropout=0.0):
        super().__init__(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            dropout=dropout
        )


def restore_metadata_as_sitk(img, source_img):
    """ Restore the metadata from the source image

    Given an image, and a source image path, load the source image and copy the
    metadata to the images in the dictionary.

    In case the img is padded, it'll be cropped according to the source image
    shape.

    :param img: Image as ndarray
    :param source_img: Reference image path
    :return: Image with restored metadata as SimpleITK image
    """
    shape = img.shape
    source_img = sitk.ReadImage(source_img) if type(source_img) == str \
        else source_img
    if len(shape) == 4:
        img_slices = []
        for i in range(shape[0]):
            img_slices.append(restore_metadata_as_sitk(img[i], source_img))
        img = sitk.JoinSeries(img_slices)
    elif len(shape) == 3:
        img = sitk.GetImageFromArray(img)
        img = sitk.PermuteAxes(img, (2, 1, 0))
        img = sitk.Flip(img, [True, True, False])  # flip the first axis
        if img.GetSize() != source_img.GetSize():
            sz = source_img.GetSize()
            img = img[0:sz[0], 0:sz[1], 0:sz[2]]
        img.CopyInformation(source_img)

    else:
        raise ValueError('Image dimension not supported')
    return img


class WMHModel(pl.LightningModule):
    """ WMHModel

    This class implements the WMH segmentation model. It is a subclass of the
    LightningModule class from PyTorch Lightning. This class is used to train
    the model and to save the predictions to the disk.

    :param net: Model instance
    :param loss: Criterion function name to use. See the get_criterion method.
    :param learning_rate: Learning rate
    :param optimizer_class: Optimizer class
    :param weight_decay: Weight decay
    :param lambda_lr: LambdaLR coefficient (scheduler)
    :param reduce_on_epoch: Reduce the learning rate every N epochs.
    """

    def __init__(self, net, criterion, learning_rate, optimizer_class,
                 weight_decay=0, lambda_lr=None, reduce_on_epoch=None,
                 meep_start=0, meep_lambda=0.3, **kwargs):
        super().__init__()

        self.lr = learning_rate
        self.net = net
        self.using_meep = False
        self.criterion = self.get_criterion(criterion, meep_start, meep_lambda)
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay
        self.lambda_lr = lambda_lr
        self.reduce_on_epoch = reduce_on_epoch
        self.start_val_log_epoch = meep_start  # start logging validation loss

        # Test-related parameters
        self.save_preds = kwargs.get('save_predictions', False)
        self.output_dir = kwargs.get('output_dir', None)
        self.saved_preds = []
        self.patch_size = kwargs.get('patch_size', None)

        self.save_hyperparameters(ignore=['net'])

    @classmethod
    def load_test(cls, model_path, save_predictions, output_dir, patch_size):
        model_path = os.path.expanduser(model_path)
        obj = WMHModel.load_from_checkpoint(model_path, net=UNet3D())
        obj.model_path = os.path.abspath(model_path)
        obj.save_preds = save_predictions
        obj.output_dir = output_dir
        obj.patch_size = patch_size
        return obj

    def get_criterion(self, loss, meep_start=0, meep_lambda=0.3):
        match loss.lower():
            case 'crossentropy' | 'ce':
                return CrossEntropyLoss()
            case 'dice':
                return DiceLoss()
            case 'dicece' | 'dicecrossentropy':
                return DiceCELoss()
            case 'focal':
                return FocalLoss()
            case 'cemeep' | 'crosentropymeep' | 'meep':
                self.using_meep = True
                return BCEMEEPLoss(meep_start, meep_lambda)
            case _:
                raise ValueError(f'Unknown loss function: {loss}')

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)
        if self.lambda_lr is not None and self.reduce_on_epoch is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: self.lambda_lr ** (
                        epoch // self.reduce_on_epoch),
                verbose=True
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

    def save_preds_tst(self, y_hat, y, logits, reference_imgs, is_test=False):
        """ Saves predictions to the disk

        It expects full volume images, not patches.

        :param y_hat: Softmax output
        :param y: Ground truth
        :param logits: Logits
        :param reference_imgs: Paths to the T1 images
        :param is_test: Whether the model is being tested or not
        """
        if not is_test or (not self.save_preds and self.output_dir is None):
            return None

        for i, t1_path in enumerate(reference_imgs):  # For each img in batch
            pred_folder = self.get_pred_folder(t1_path)
            hard_pred = torch.argmax(y_hat[i], dim=0).to(torch.uint8)

            run_id = os.path.splitext(os.path.basename(self.model_path))[0]
            imgs = {
                f'pred_wmh_hard_{run_id}.nii.gz': hard_pred,
                f'pred_wmh_softmax_{run_id}.nii.gz': y_hat[i],
                f'pred_logits_{run_id}.nii.gz': logits[i],
                f'gt_wmh_{run_id}.nii.gz':
                    y[i, 1].to(torch.uint8) if y is not None else None,
            }

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

        # Concatenate the input images along the channel dimension
        x = torch.cat((xc1, xc2), dim=1)

        # Forward pass through the neural network
        logits = self.forward_pass(x, batch, is_test)
        y_hat = F.softmax(logits, dim=1)

        self.save_preds_tst(y_hat, y, logits, t1_paths, is_test)

        return y_hat, y  # Return the predicted and GT masks

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y, epoch) if self.using_meep \
            else self.criterion(y_hat, y)
        metrics = compute_metrics(y_hat, y, text='train_')

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        epoch = self.current_epoch
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y, epoch) if self.using_meep \
            else self.criterion(y_hat, y)
        metrics = compute_metrics(y_hat, y, text='val_')

        self.log('val_loss', loss, on_step=True, on_epoch=True)

        loss_after_n = torch.inf if epoch < self.start_val_log_epoch else loss
        self.log('val_loss_after_n', loss_after_n, on_epoch=True)

        self.log_dict(metrics, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        self.infer_batch(batch, is_test=True)

    def on_train_epoch_end(self):  # Todo change filtering _epoch
        metr, ce = self.trainer.callback_metrics, self.current_epoch
        mlflow.log_metric('train_loss', metr['train_loss_epoch'].item(), ce)
        mlflow.log_metric('train_dice', metr['train_dice_epoch'].item(), ce)

    def on_validation_epoch_end(self):
        metr, ce = self.trainer.callback_metrics, self.current_epoch
        mlflow.log_metric('val_loss', metr['val_loss_epoch'].item(), ce)
        mlflow.log_metric('val_dice', metr['val_dice_epoch'].item(), ce)

    def save_preds_info(self, preds_info_path, model_path, centers):
        """ Saves the paths to the saved predictions

        :param preds_info_path: Path to the file where the paths will be saved
        :param model_path: Path to the model checkpoint
        :param centers: List of centers used for testing
        """
        if preds_info_path is None:
            folder_name = '_'.join(
                os.path.basename(model_path).split('_')[0:-1])
            file_name = os.path.basename(model_path).replace('.ckpt', '.csv')
            preds_info_path = os.path.join(os.getcwd(), 'notebooks',
                                           folder_name, file_name)

        preds_info_path = os.path.abspath(os.path.expanduser(preds_info_path))
        os.makedirs(os.path.dirname(preds_info_path), exist_ok=True)

        with open(preds_info_path, 'w', ) as f:
            writer = csv.writer(f)
            writer.writerows(self.saved_preds)
        print(f'Predictions info saved to {preds_info_path}')
