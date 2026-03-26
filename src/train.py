import ast
import os.path
import random
import time
from datetime import datetime

import click
import lightning as L
import mlflow.sklearn
import torch
from mlflow import MlflowClient

from datamodules import WMHDataModule
from models.unet3d import UNet3D
from models.wmh_module import WMHModel
from utils.cli import load_defaults
from utils.metrics import compute_metrics

print("Last run on", time.ctime())


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(
        f"run_id: {r.info.run_id}\n"
        f"artifacts: {artifacts}\n"
        f"params: {r.data.params}\n"
        f"metrics: {compute_metrics}\n"
        f"tags: {tags}"
    )


@click.command(context_settings=dict(default_map=load_defaults("main")))
@click.option("--data-root", type=click.STRING, required=True, help="Root data folder")
@click.option(
    "--centers",
    type=click.STRING,
    required=True,
    help="Centers used for training (e.g.: training:Utretch)",
)
@click.option(
    "--split-ratios",
    type=click.STRING,
    required=True,
    help="Ratios for training/validation/test splits",
)
@click.option("--epochs", required=True, type=click.INT, help="Number of epochs to train the model")
@click.option(
    "--batch-size", required=True, type=click.INT, help="Batch size to use during training"
)
@click.option("--lr", required=True, type=click.FLOAT, help="Learning rate for optimizer")
@click.option(
    "--dropout", required=True, type=click.FLOAT, help="Dropout probability for the model"
)
@click.option("--loss", required=True, type=click.STRING, help="Loss function to use")
@click.option("--weight-decay", required=True, type=click.FLOAT, help="Weight decay for optimizer")
@click.option("--seed", required=True, type=click.INT, help="Random seed for reproducibility")
@click.option(
    "--patch-size", required=True, type=click.types.INT, help="Patch size to use for training"
)
@click.option(
    "--samples-per-volume",
    required=True,
    type=click.INT,
    help="Number of patches to sample per volume",
)
@click.option("--queue-length", required=True, type=click.INT, help="Patch queue length")
@click.option("--tio-num-workers", required=True, type=click.INT, help="Patch queue length")
@click.option("--custom-name", required=False, type=click.STRING, help="Custom name for the run")
@click.option("--resume-from", required=False, type=click.STRING, help="Resume from checkpoint")
@click.option("--lambda-lr", required=False, type=click.FLOAT, help="LambdaLR Scheduler")
@click.option(
    "--reduce-on-epoch", required=False, type=click.INT, help="When to reduce LR (scheduler)"
)
@click.option(
    "--reg-start", required=False, type=click.INT, help="When to start adding regularization term"
)
@click.option(
    "--meep-lambda",
    required=False,
    type=click.FLOAT,
    help="Lambda for MEEP (if it was selected a loss that includes it).",
)
@click.option(
    "--ood-centers",
    required=False,
    type=click.STRING,
    help="Centers used as out of distribution data, used for MEOOD "
    "loss, separated by commas (e.g.: Utretch,Singapore)",
)
def train(
    data_root,
    centers,
    split_ratios,
    epochs,
    batch_size,
    lr,
    dropout,
    loss,
    weight_decay,
    seed,
    patch_size,
    samples_per_volume,
    queue_length,
    tio_num_workers,
    custom_name,
    resume_from,
    lambda_lr,
    reduce_on_epoch,
    reg_start,
    meep_lambda,
    ood_centers,
):
    split_ratios = ast.literal_eval(split_ratios)
    patch_size = None if patch_size == -1 else patch_size
    resume_from = None if resume_from == "None" else resume_from
    params = {
        "data_root": data_root,
        "centers": centers,
        "split_ratios": split_ratios,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dropout": dropout,
        "loss": loss,
        "weight_decay": weight_decay,
        "seed": seed,
        "patch_size": patch_size,
        "samples_per_volume": samples_per_volume,
        "queue_length": queue_length,
        "tio_num_workers": tio_num_workers,
        "custom_name": custom_name,
        "resume_from": resume_from,
        "lambda_lr": lambda_lr,
        "reduce_on_epoch": reduce_on_epoch,
        "reg_start": reg_start,
        "meep_lambda": meep_lambda,
        "ood_centers": ood_centers,
    }

    dataloader = WMHDataModule(
        data_root,
        batch_size,
        centers,
        split_ratios,
        patch_size,
        seed,
        tio_num_workers,
        samples_per_volume,
        queue_length,
    )

    run_name = centers.replace(":", "_").replace(",", "_")
    run_name += f"_{loss}_{random.randint(1000, 9999)}"
    run_name = custom_name if custom_name else run_name

    params["run_name"] = run_name

    with mlflow.start_run(run_name=run_name) as run:
        if len(run.data.params) == 0:
            mlflow.log_params(params)

        best_model_path = os.path.join("checkpoints", f"{run_name}_best.ckpt")
        top3_chk = L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join("checkpoints", run_name),
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
        )

        trainer = L.Trainer(
            accelerator="auto",
            max_epochs=epochs,
            callbacks=[top3_chk],
            devices="auto",
        )

        net = UNet3D(dropout=dropout)

        model = WMHModel(
            net=net,
            criterion=loss,
            learning_rate=lr,
            optimizer_class=torch.optim.AdamW,
            weight_decay=weight_decay,
            lambda_lr=lambda_lr,
            reduce_on_epoch=reduce_on_epoch,
            reg_start=reg_start,
            reg_lambda=meep_lambda,
            best_model_path=best_model_path,
            ood_centers=ood_centers,
        )

        start = datetime.now()
        print("Training started at", start)
        trainer.fit(model, dataloader, ckpt_path=resume_from)

        print("Training duration:", datetime.now() - start)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    if os.getcwd().endswith("src"):
        os.chdir("..")
    train()
