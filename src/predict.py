import time

import click
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchio as tio

from common import WMHModel
from datamodules import WMHDataModule

print('Last run on', time.ctime())


def prepare_batch(batch):
    return batch['t1'][tio.DATA], batch['flair'][tio.DATA], \
        batch['wmh'][tio.DATA]


def infer_batch(net, batch):
    xc1, xc2, y = prepare_batch(batch)
    x = torch.cat((xc1, xc2), dim=1)  # Concatenate the channels
    y_hat = F.softmax(net(x), dim=1)
    return y_hat


@click.command()
@click.option('--data-root', type=click.STRING, required=True,
              help="Root data folder")
@click.option('--centers', type=click.STRING, required=True,
              help="Centers used for training (e.g.: training:Utretch)")
@click.option('--split-ratios', type=click.STRING, required=True,
              help="Ratios for training/validation/test splits")
@click.option('--model-path', type=click.STRING, required=True,
              help="Path to the trained model weights")
@click.option('--output-path', type=click.STRING, required=True,
              help="Path to the output file where predictions will be saved")
@click.option('--batch-size', type=click.INT, default=1,
              help='Batch size to use during inference')
@click.option('--tio-num-workers', required=True, type=click.INT,
              help='Number of workers to use for TorchIO DataLoader')
@click.option('--seed', required=True, type=click.INT,
              help='Random seed for reproducibility')
@click.option('--patch-size', required=True, type=click.INT,
              help='Patch size to use for training')
def predict(data_root, centers, split_ratios, model_path, output_path,
            batch_size, tio_num_workers, seed, patch_size):
    dataloader = WMHDataModule(data_root, batch_size, centers, split_ratios,
                               patch_size, seed, tio_num_workers)

    # or call with pretrained model
    model = WMHModel.load_from_checkpoint(model_path)
    trainer = pl.Trainer()
    trainer.test(model, dataloader)


if __name__ == '__main__':
    predict()
