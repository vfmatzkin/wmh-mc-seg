import ast
import os.path
import shutil
import sys
import time
from datetime import datetime

import click
import mlflow.sklearn
import monai
import pytorch_lightning as pl
import torch
from mlflow import MlflowClient

from model import WMHModel, compute_metrics
from datamodules import WMHDataModule

print('Last run on', time.ctime())


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if
            not k.startswith("mlflow.")}
    artifacts = [f.path for f in
                 MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}\n"
          f"artifacts: {artifacts}\n"
          f"params: {r.data.params}\n"
          f"metrics: {compute_metrics}\n"
          f"tags: {tags}")


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
@click.option('--dropout', required=True, type=click.FLOAT,
              help='Dropout probability for the model')
@click.option('--weight-decay', required=True, type=click.FLOAT,
              help='Weight decay for optimizer')
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
@click.option('--custom-name', required=False, type=click.STRING,
              help='Custom name for the run')
def train(data_root, centers, split_ratios, epochs, batch_size, lr, dropout,
          weight_decay, seed, patch_size, samples_per_volume, queue_length,
          tio_num_workers, custom_name):
    split_ratios = ast.literal_eval(split_ratios)

    dataloader = WMHDataModule(data_root, batch_size, centers, split_ratios,
                               patch_size, seed, tio_num_workers,
                               samples_per_volume, queue_length)

    run_name = custom_name if custom_name else centers

    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        mlflow.set_tag('run_name', run_name)

        top3_chk = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join('checkpoints', run_name),
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode='min',
        )

        trainer = pl.Trainer(
            accelerator='auto',
            max_epochs=epochs,
            callbacks=[top3_chk],
            devices='auto'
        )

        unet = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            dropout=dropout,
        )

        model = WMHModel(
            net=unet,
            criterion='DiceCE',
            learning_rate=lr,
            optimizer_class=torch.optim.AdamW,
            weight_decay=weight_decay,
        )

        start = datetime.now()
        print('Training started at', start)
        trainer.fit(model, dataloader)

        # Save best model
        best_model_path = os.path.join('checkpoints', f'{run_name}_best.ckpt')
        shutil.copy(top3_chk.best_model_path, best_model_path)

        print('Training duration:', datetime.now() - start)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train()
    else:
        mlflow.projects.run('..', env_manager='local')
