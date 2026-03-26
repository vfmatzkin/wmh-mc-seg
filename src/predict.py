import ast
import os
import time

import click
import pytorch_lightning as pl

from datamodules import WMHDataModule
from model import WMHModel
from utils.cli import load_defaults

print('Last run on', time.ctime())


@click.command(context_settings=dict(default_map=load_defaults('test')))
@click.option('--data-root', type=click.STRING, required=True,
              help="Root data folder")
@click.option('--centers', type=click.STRING, required=True,
              help="Centers used for training (e.g.: training:Utrecht)")
@click.option('--split-ratios', type=click.STRING, required=True,
              help="Ratios for training/validation/test splits")
@click.option('--model-path', type=click.STRING, required=True,
              help="Path to the trained model weights")
@click.option('--batch-size', type=click.INT, default=1,
              help='Batch size to use during inference')
@click.option('--patch-size', type=click.INT, default=None,
              help='Patch size to use for prediction')
@click.option('--seed', required=True, type=click.INT,
              help='Random seed for reproducibility')
@click.option('--output-dir', type=click.STRING, default=None,
              help='Output directory for the predictions')
@click.option('--save-predictions/--no-save-predictions', type=click.BOOL,
              default=True, help='Save predictions along with the Ground Truth '
                                 'images if output-dir is not provided')
@click.option('--csv-preds', type=click.STRING, default=None,
              help='CSV file to save the predictions paths')
@click.option('--mc-ratio', type=click.FLOAT, default=0.0,
              help='Dropout ratio for Monte Carlo Dropout. Note that it has to'
              ' be lower than the dropout used for training.')
@click.option('--mc-samples', type=click.INT, default=0,
              help='Number of Monte Carlo Dropout samples')
@click.option('--predict-split', type=click.STRING, default='test',
              help='Split to use for prediction')
def predict(data_root, centers, split_ratios, model_path, batch_size,
            patch_size, seed, output_dir, save_predictions, csv_preds,
            mc_ratio, mc_samples, predict_split):
    split_ratios = ast.literal_eval(split_ratios)
    patch_size = None if patch_size == -1 else patch_size

    dataloader = WMHDataModule(data_root, batch_size, centers, split_ratios,
                               seed=seed, predict_split=predict_split)

    model = WMHModel.load_test(model_path, save_predictions, output_dir,
                               patch_size, mc_ratio, mc_samples)

    trainer = pl.Trainer()
    trainer.test(model, dataloader)

    model.save_preds_info(csv_preds, model_path)


if __name__ == '__main__':
    if os.getcwd().endswith('src'):
        os.chdir('..')
    predict()


