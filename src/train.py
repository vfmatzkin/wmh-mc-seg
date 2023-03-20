import time

import click
import mlflow.sklearn
import monai
import pytorch_lightning as pl
import torch
import torchio as tio
from mlflow import MlflowClient

from datamodules import WMHDataModule

print('Last run on', time.ctime())


class WMHModel(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['t1'][tio.DATA], batch['flair'][tio.DATA], \
            batch['wmh'][tio.DATA]

    def infer_batch(self, batch):
        xc1, xc2, y = self.prepare_batch(batch)
        x = torch.cat((xc1, xc2), dim=1)  # Concatenate the channels
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if
            not k.startswith("mlflow.")}
    artifacts = [f.path for f in
                 MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


@click.option('--data-root', type=click.STRING, required=True,
              help="Root data folder")
@click.option('--centers', type=click.STRING, required=True,
              help="Centers used for training (e.g.: training:Utretch)")
@click.option('--train-ratio', type=click.INT, required=True,
              help="Ratio of training data to use for training")
@click.option('--epochs', required=True, type=click.INT,
              help='Number of epochs to train the model')
@click.option('--batch-size', required=True, type=click.INT,
              help='Batch size to use during training')
@click.option('--lr', required=True, type=click.FLOAT
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
def main(data_root, centers, train_ratio, epochs, batch_size, lr, weight_decay,
         losses, seed, patch_size):
    dataloader = WMHDataModule(data_root, batch_size, centers, train_ratio,
                               seed)
    early_stopping = pl.callbacks.early_stopping.EarlyStopping('val_loss')
    trainer = pl.Trainer(
        max_epochs=epochs,
        progress_bar_refresh_rate=20,
        callbacks=[early_stopping],
        # gpus=1 if torch.cuda.is_available() else 0,
        # precision=16,
    )

    mlflow.pytorch.autolog()
    with mlflow.start_run():
        unet = monai.networks.nets.UNet(
            dimensions=3,
            in_channels=2,
            out_channels=3,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
        )

        model = WMHModel(
            net=unet,
            criterion=monai.losses.DiceCELoss(softmax=True),
            learning_rate=lr,
            optimizer_class=torch.optim.AdamW,
        )

        trainer.fit(model, dataloader)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    main()
