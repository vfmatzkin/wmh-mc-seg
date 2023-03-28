import sys
import time
from datetime import datetime

import click
import mlflow.sklearn
import monai
import pytorch_lightning as pl
import torch
import torchio as tio
from mlflow import MlflowClient
import tempfile

from datamodules import WMHDataModule

print('Last run on', time.ctime())


def prevent_logging(debug=True, force=False):
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
        text + 'dice': monai.metrics.compute_meandice(y_hat, y)
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

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def infer_batch(self, batch):
        xc1, xc2, y = prepare_batch(batch)
        x = torch.cat((xc1, xc2), dim=1)  # Concatenate the channels
        y_hat = self.net(x)
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


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if
            not k.startswith("mlflow.")}
    artifacts = [f.path for f in
                 MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.compute_metrics))
    print("tags: {}".format(tags))


@click.command()
@click.option('--data-root', type=click.STRING, required=True,
              help="Root data folder")
@click.option('--centers', type=click.STRING, required=True,
              help="Centers used for training (e.g.: training:Utretch)")
@click.option('--split-ratios', type=click.FLOAT, required=True, nargs=3,
              help="Ratios for training/validation/test splits")
@click.option('--epochs', required=True, type=click.INT,
              help='Number of epochs to train the model')
@click.option('--batch-size', required=True, type=click.INT,
              help='Batch size to use during training')
@click.option('--lr', required=True, type=click.FLOAT,
              help='Learning rate for optimizer')
@click.option('--weight-decay', required=True, type=click.FLOAT,
              help='Weight decay for optimizer')
@click.option('--losses', required=True, type=click.STRING,
              help='List of losses to use with their respective lambda values,'
                   ' e.g. "DiceLoss,0.5,MSELoss,0.5"')
@click.option('--seed', required=True, type=click.INT,
              help='Random seed for reproducibility')
@click.option('--patch-size', required=True, type=click.INT,
              help='Patch size to use for training')
@click.option('--samples-per-volume', required=True, type=click.INT,
              help='Number of patches to sample per volume')
@click.option('--queue-length', required=True, type=click.INT,
              help='Patch queue length')
@click.option('--tio-num-workers', required=True, type=click.INT,
              help='Patch queue length')
@click.option('--disable-logging', required=True, type=click.BOOL,
              help='Force disable MLFlow logging')
def main(data_root, centers, split_ratios, epochs, batch_size, lr, weight_decay,
         losses, seed, patch_size, samples_per_volume, queue_length,
         tio_num_workers, disable_logging):
    prevent_logging(debug=True, force=disable_logging)
    dataloader = WMHDataModule(data_root, batch_size, centers, split_ratios,
                               patch_size, seed, samples_per_volume,
                               queue_length, tio_num_workers)

    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        val_chk = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename=f'{run.info.run_name}--'"{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode='min',
        )

        trainer = pl.Trainer(
            accelerator='gpu',
            max_epochs=epochs,
            callbacks=[val_chk],
            gpus=1 if torch.cuda.is_available() else 0,
        )

        unet = monai.networks.nets.UNet(
            dimensions=3,
            in_channels=2,
            out_channels=2,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
        )

        model = WMHModel(
            net=unet,
            criterion=monai.losses.DiceCELoss(softmax=True),
            learning_rate=lr,
            optimizer_class=torch.optim.AdamW,
        )

        start = datetime.now()
        print('Training started at', start)
        trainer.fit(model, dataloader)
        print('Training duration:', datetime.now() - start)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    main()
