"""Pure inference functions for WMH segmentation.

These functions take the network (and other config) as explicit parameters
instead of using self, keeping them decoupled from the LightningModule.
"""

from __future__ import annotations

import os

import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio

from utils.sitk_io import restore_metadata_as_sitk


def get_pred_folder(t1_path: str, output_dir: str | None) -> str:
    """Return (and create) the folder where predictions for a subject will be saved.

    :param t1_path: Path to the T1 image for this subject
    :param output_dir: Base output directory, or None to write next to the source
    """
    parent_folder = os.path.dirname(os.path.dirname(t1_path))
    if output_dir is None:
        pred_folder = parent_folder
    else:
        pred_folder = os.path.join(output_dir, os.path.basename(parent_folder))
    os.makedirs(pred_folder, exist_ok=True)
    return pred_folder


def patch_inference(
    net: torch.nn.Module,
    x: torch.Tensor,
    batch: dict,
    patch_size: int,
) -> torch.Tensor:
    """Run inference by splitting each subject into patches and re-assembling.

    :param net: The segmentation network
    :param x: Input tensor (B, C, D, H, W) — kept for shape reference but
              the actual patch data is read from batch
    :param batch: TorchIO batch dict with 't1', 'flair', and optionally 'wmh'
    :param patch_size: Patch size passed to GridSampler
    """
    out_tensor = torch.empty((0, x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
    for i_subj in range(x.shape[0]):
        if "wmh" in batch:
            subject = tio.Subject(
                t1=tio.ScalarImage(tensor=batch["t1"]["data"][i_subj]),
                flair=tio.ScalarImage(tensor=batch["flair"]["data"][i_subj]),
                wmh=tio.LabelMap(tensor=batch["wmh"]["data"][i_subj]),
            )
        else:
            subject = tio.Subject(
                t1=tio.ScalarImage(tensor=batch["t1"]["data"][i_subj]),
                flair=tio.ScalarImage(tensor=batch["flair"]["data"][i_subj]),
            )
        grid_sampler = tio.inference.GridSampler(subject, patch_size, 4)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler)
        with torch.no_grad():
            for patches_batch in patch_loader:
                xc1 = patches_batch["t1"][tio.DATA]
                xc2 = patches_batch["flair"][tio.DATA]
                x_patch = torch.cat((xc1, xc2), dim=1)
                locations = patches_batch[tio.LOCATION]
                logits = net(x_patch)
                aggregator.add_batch(logits, locations)
        output_tensor = aggregator.get_output_tensor()
        out_tensor = torch.cat((out_tensor, output_tensor.unsqueeze(0)))
    return out_tensor


def forward_pass(
    net: torch.nn.Module,
    x: torch.Tensor,
    batch: dict,
    is_test: bool,
    patch_size: int | None,
) -> torch.Tensor:
    """Single forward pass, with optional patch-based inference.

    :param net: The segmentation network
    :param x: Input tensor
    :param batch: TorchIO batch dict
    :param is_test: Whether we are in test mode
    :param patch_size: If set and is_test is True, use patch inference
    """
    if not is_test or patch_size is None:
        return net(x)
    return patch_inference(net, x, batch, patch_size)


def get_mc_preds(
    net: torch.nn.Module,
    x: torch.Tensor,
    batch: dict,
    mc_dropout_ratio: float,
    mc_dropout_samples: int,
    patch_size: int | None,
    is_test: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MC dropout forward passes and return logits + softmax arrays.

    :param net: The segmentation network
    :param x: Input tensor
    :param batch: TorchIO batch dict
    :param mc_dropout_ratio: Dropout ratio used in the network
    :param mc_dropout_samples: Number of stochastic forward passes
    :param patch_size: Passed through to forward_pass
    :param is_test: Whether we are in test mode
    :returns: (logits_arr, y_hat_arr) — empty tensors when MC is disabled
    """
    mc_cond = mc_dropout_ratio > 0 and is_test
    if mc_cond:
        model_was_eval = not net.training
        net.train()

    mc_fwd_passes = mc_dropout_samples if mc_cond else 0
    logits_arr = torch.empty((mc_fwd_passes, x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4]))
    y_hat_arr = torch.empty((mc_fwd_passes, x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4]))

    for i in range(mc_fwd_passes):
        print(f"  - MC dropout: forward pass {i + 1}/{mc_fwd_passes}")
        logits_mc = forward_pass(net, x, batch, is_test, patch_size)
        y_hat_mc = F.softmax(logits_mc, dim=1)
        logits_arr[i] = logits_mc
        y_hat_arr[i] = y_hat_mc

    if mc_cond and model_was_eval:
        net.eval()

    return logits_arr, y_hat_arr


def save_predictions(
    y_hat: torch.Tensor,
    y: torch.Tensor | None,
    logits: torch.Tensor,
    reference_imgs: list[str],
    model_path: str,
    output_dir: str | None,
    save_preds: bool,
    mc_dropout_samples: int,
    is_test: bool = False,
    lgs_mc_arr: torch.Tensor | None = None,
    sm_mc_arr: torch.Tensor | None = None,
) -> list[list[str]] | None:
    """Save predictions (and optional MC dropout outputs) to disk.

    Expects full-volume images, not patches.

    :param y_hat: Softmax output
    :param y: Ground truth (or None)
    :param logits: Logits
    :param reference_imgs: Paths to the T1 images (one per batch item)
    :param model_path: Path to the loaded checkpoint (used for run_id in filenames)
    :param output_dir: Base output directory, or None
    :param save_preds: Whether saving is enabled
    :param mc_dropout_samples: Number of MC samples (used for uncertainty guard)
    :param is_test: Only saves when True
    :param lgs_mc_arr: Logits array from MC passes (may be empty)
    :param sm_mc_arr: Softmax array from MC passes (may be empty)
    :returns: List of saved paths per subject, or None when saving is skipped
    """
    if not is_test or (not save_preds and output_dir is None):
        return None

    saved = []
    for i, t1_path in enumerate(reference_imgs):
        pred_folder = get_pred_folder(t1_path, output_dir)
        hard_pred = torch.argmax(y_hat[i], dim=0).to(torch.uint8)

        lgs_mc_mean = lgs_mc_arr.mean(0) if lgs_mc_arr is not None else None
        sftmx_mc_mean = sm_mc_arr.mean(0) if sm_mc_arr is not None else None
        hard_pred_mc, uncert_mc = None, None
        if sm_mc_arr is not None:
            hard_pred_mc = torch.argmax(sftmx_mc_mean[0], 0).to(torch.uint8)
            if mc_dropout_samples == 1:
                print("WARNING: MC dropout samples == 1, uncertainty estimation won't be computed")
            uncert_mc = (
                sm_mc_arr.std(0)[0] if mc_dropout_samples > 1 else None
            )  # First elem in batch + foreground ch on save

        run_id = os.path.splitext(os.path.basename(model_path))[0]
        imgs = {
            f"pred_wmh_hard_{run_id}.nii.gz": hard_pred,
            f"pred_wmh_softmax_{run_id}.nii.gz": y_hat[i],
            f"pred_logits_{run_id}.nii.gz": logits[i],
            f"gt_wmh_{run_id}.nii.gz": (y[i, 1].to(torch.uint8) if y is not None else None),
        }

        if lgs_mc_arr is not None and lgs_mc_arr.shape[0]:
            imgs.update(
                {
                    f"pred_mc_logitsmean_{run_id}.nii.gz": lgs_mc_mean[0],
                    f"pred_mc_softmaxmean_{run_id}.nii.gz": sftmx_mc_mean[0],
                    f"pred_mc_hardmean_{run_id}.nii.gz": hard_pred_mc,
                }
            )

        if uncert_mc is not None:
            imgs.update({f"pred_mc_uncertmc_{run_id}.nii.gz": uncert_mc[1]})

        paths = []
        for img_name, img in imgs.items():
            paths.append(os.path.join(pred_folder, img_name))
            if img is not None:
                im_o_sp = restore_metadata_as_sitk(img, t1_path)
                sitk.WriteImage(im_o_sp, paths[-1])
        saved.append(paths)
        print(f"saved preds for {pred_folder}")

    return saved
