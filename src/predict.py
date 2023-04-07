import ast
import time

import click
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchio as tio

from src.model import WMHModel
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
@click.option('--run-id', type=click.STRING, required=True,
              help="Run ID to identify the model predictions for that run")
@click.option('--batch-size', type=click.INT, default=1,
              help='Batch size to use during inference')
@click.option('--tio-num-workers', required=True, type=click.INT,
              help='Number of workers to use for TorchIO DataLoader')
@click.option('--seed', required=True, type=click.INT,
              help='Random seed for reproducibility')
@click.option('--patch-size', required=True, type=click.INT,
              help='Patch size to use for training')
@click.option('--output-dir', type=click.STRING, default=None,
              help='Output directory for the predictions')
@click.option('--save-predictions/--no-save-predictions', type=click.BOOL,
              default=True, help='Save predictions along with the Ground Truth '
                                 'images if output-dir is not provided')
@click.option('--csv-preds', type=click.STRING, default=None,
              help='CSV file to save the predictions paths')
def predict(data_root, centers, split_ratios, model_path, run_id,
            batch_size, tio_num_workers, seed, patch_size, output_dir,
            save_predictions, csv_preds):
    split_ratios = ast.literal_eval(split_ratios)
    csv_preds = f'predictions_{run_id}.csv' if csv_preds is None else csv_preds

    dataloader = WMHDataModule(data_root, batch_size, centers, split_ratios,
                               patch_size, seed, tio_num_workers)

    model = WMHModel.load_from_checkpoint(model_path)
    model.run_id = run_id
    model.save_preds = save_predictions
    model.output_dir = output_dir

    trainer = pl.Trainer()
    trainer.test(model, dataloader)

    model.save_preds_info(csv_preds)


if __name__ == '__main__':
    predict()
