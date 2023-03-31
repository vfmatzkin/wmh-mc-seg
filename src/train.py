import ast
import os.path
import time
from datetime import datetime

import click
import mlflow.sklearn
import monai
import pytorch_lightning as pl
import torch
from mlflow import MlflowClient

from common import prevent_logging, WMHModel
from datamodules import WMHDataModule

print('Last run on', time.ctime())


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
@click.option('--split-ratios', type=click.STRING, required=True,
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
def train(data_root, centers, split_ratios, epochs, batch_size, lr,
          weight_decay, losses, seed, patch_size, samples_per_volume,
          queue_length, tio_num_workers, disable_logging):
    split_ratios = ast.literal_eval(split_ratios)

    prevent_logging(disable_logging)
    dataloader = WMHDataModule(data_root, batch_size, centers, split_ratios,
                               patch_size, seed, tio_num_workers,
                               samples_per_volume, queue_length)

    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        val_chk = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join('checkpoints', run.info.run_name),
            filename="{epoch:02d}-{val_loss:.3f}",
            save_top_k=3,
            mode='min',
        )

        trainer = pl.Trainer(
            accelerator='auto',
            max_epochs=epochs,
            callbacks=[val_chk],
            devices='auto'
        )

        unet = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
        )

        model = WMHModel(
            net=unet,
            criterion=monai.losses.DiceCELoss(),
            learning_rate=lr,
            optimizer_class=torch.optim.AdamW,
        )

        start = datetime.now()
        print('Training started at', start)
        trainer.fit(model, dataloader)
        print('Training duration:', datetime.now() - start)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    train()
