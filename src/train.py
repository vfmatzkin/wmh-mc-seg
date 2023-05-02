import ast
import os.path
import random
import shutil
import sys
import time
from datetime import datetime

import click
import mlflow.sklearn
import pytorch_lightning as pl
import torch
import yaml
from mlflow import MlflowClient

from datamodules import WMHDataModule
from model import WMHModel, compute_metrics, UNet3D

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
@click.option('--patch-size', required=True, type=click.types.INT,
              help='Patch size to use for training')
@click.option('--samples-per-volume', required=True, type=click.INT,
              help='Number of patches to sample per volume')
@click.option('--queue-length', required=True, type=click.INT,
              help='Patch queue length')
@click.option('--tio-num-workers', required=True, type=click.INT,
              help='Patch queue length')
@click.option('--custom-name', required=False, type=click.STRING,
              help='Custom name for the run')
@click.option('--resume-from', required=False, type=click.STRING,
              help='Resume from checkpoint')
def train(data_root, centers, split_ratios, epochs, batch_size, lr, dropout,
          weight_decay, seed, patch_size, samples_per_volume, queue_length,
          tio_num_workers, custom_name, resume_from):
    split_ratios = ast.literal_eval(split_ratios)
    patch_size = None if patch_size == -1 else patch_size
    resume_from = None if resume_from == 'None' else resume_from
    params = {
        'data_root': data_root,
        'centers': centers,
        'split_ratios': split_ratios,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'dropout': dropout,
        'weight_decay': weight_decay,
        'seed': seed,
        'patch_size': patch_size,
        'samples_per_volume': samples_per_volume,
        'queue_length': queue_length,
        'tio_num_workers': tio_num_workers,
        'custom_name': custom_name,
        'resume_from': resume_from,
    }

    dataloader = WMHDataModule(data_root, batch_size, centers, split_ratios,
                               patch_size, seed, tio_num_workers,
                               samples_per_volume, queue_length)

    run_name = custom_name if custom_name \
        else centers.replace(':', '_').replace(',', '_')
    run_name_o = run_name
    run_name += f'_{random.randint(1000, 9999)}'

    with mlflow.start_run(run_name=run_name) as run:
        if len(run.data.params) == 0:
            mlflow.log_params(params)

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
            devices='auto',
        )

        net = UNet3D(dropout=dropout)

        model = WMHModel(
            net=net,
            criterion='DiceCE',
            learning_rate=lr,
            optimizer_class=torch.optim.AdamW,
            weight_decay=weight_decay,
        )

        start = datetime.now()
        print('Training started at', start)
        trainer.fit(model, dataloader, ckpt_path=resume_from)

        # Save best model
        best_model_path = os.path.join('checkpoints', f'{run_name_o}_best.ckpt')
        if not os.path.exists(top3_chk.best_model_path):
            print(f'Best model path does not exist: {top3_chk.best_model_path}.'
                  f' This may happen when resuming from a checkpoint, and the '
                  f'best model is still the one from the previous run. ')
            if os.path.exists(resume_from):
                print(f'Copying previous checkpoint from {resume_from} to '
                      f'{best_model_path} as best model.')
            shutil.copy(resume_from, best_model_path)
        else:
            shutil.copy(top3_chk.best_model_path, best_model_path)

        print('Training duration:', datetime.now() - start)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    if os.getcwd().endswith('src'):
        os.chdir('..')

    if len(sys.argv) == 1:
        with open('MLproject', 'r') as f:
            mlproject = yaml.safe_load(f)
        params = mlproject['entry_points']['main']['parameters']
        sys.argv += [f"--{k.replace('_', '-')}="
                     f"{v['default']}" for k, v in params.items()]
    train()
