import csv
import os

import SimpleITK as sitk
import monai
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


class DiceCE(torch.nn.Module):
    def __init__(self):
        """ DiceCE

        This class implements the DiceCE loss function. It is a subclass of the
        torch.nn.Module class.
        This is an example for creating combination of losses. It's important
        to log the loss name and the custom parameters in case of different
        values.
        """
        super().__init__()
        self.DiceCE = monai.losses.DiceCELoss()

    def forward(self, y_pred, y_true):
        return self.DiceCE(y_pred, y_true)


class WMHModel(pl.LightningModule):
    """ WMHModel

    This class implements the WMH segmentation model. It is a subclass of the
    LightningModule class from PyTorch Lightning. This class is used to train
    the model and to save the predictions to the disk.

    :param net: Model instance
    :param loss: Criterion function name to use. See the get_criterion method.
    :param learning_rate: Learning rate
    :param optimizer_class: Optimizer class
    """

    def __init__(self, net, criterion, learning_rate, optimizer_class,
                 weight_decay=0, **kwargs):
        super().__init__()

        self.lr = learning_rate
        self.net = net
        self.criterion = self.get_criterion(criterion)
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay

        # Test-related parameters
        self.save_preds = kwargs.get('save_predictions', False)
        self.output_dir = kwargs.get('output_dir', None)
        self.saved_preds = []

        # TODO Remove when removing autolog
        self.save_hyperparameters(ignore=['net', 'criterion'])

    @classmethod
    def load_test(cls, model_path, save_predictions, output_dir):
        model_path = os.path.abspath(model_path)
        obj = WMHModel.load_from_checkpoint(model_path)
        obj.model_path = os.path.abspath(model_path)
        obj.save_preds = save_predictions
        obj.output_dir = output_dir
        return obj

    def get_criterion(self, loss):
        if loss == 'DiceCE':
            return DiceCE()
        else:
            raise ValueError(f'Unknown loss function: {loss}')

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)
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
        self.infer_batch(batch, save_preds=True)

    def save_preds_info(self, preds_info_path, model_path, centers):
        """ Saves the paths to the saved predictions

        :param preds_info_path: Path to the file where the paths will be saved
        :param model_path: Path to the model checkpoint
        :param centers: List of centers used for testing
        """
        if preds_info_path is None:
            preds_info_path = model_path.replace('.ckpt', f'_{centers}.csv')
        with open(preds_info_path, 'w', ) as f:
            writer = csv.writer(f)
            writer.writerows(self.saved_preds)
        print(f'Predictions info saved to {preds_info_path}')
