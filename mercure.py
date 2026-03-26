"""Mercure processing module for WMH segmentation.

Reads T1 + FLAIR NIfTI from MERCURE_IN_DIR, runs inference with
MC dropout uncertainty, writes results to MERCURE_OUT_DIR.
"""

import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def run():
    in_dir = Path(os.environ.get("MERCURE_IN_DIR", "/tmp/data"))
    out_dir = Path(os.environ.get("MERCURE_OUT_DIR", "/tmp/output"))

    print(f"wmh-segmentation: processing from {in_dir}")

    # Find T1 and FLAIR volumes
    t1_path, flair_path = _find_inputs(in_dir)
    if t1_path is None or flair_path is None:
        _write_result(out_dir, "failed", error="Need both T1 and FLAIR NIfTI files")
        return

    try:
        outputs = run_inference(t1_path, flair_path, out_dir)
        _write_result(out_dir, "completed", outputs=outputs)
    except Exception as e:
        print(f"  Error: {e}")
        _write_result(out_dir, "failed", error=str(e))


def run_inference(
    t1_path: Path, flair_path: Path, output_dir: Path,
    checkpoint: str = "checkpoints/training_Utrecht_best.ckpt",
    mc_samples: int = 10, mc_ratio: float = 0.2,
) -> dict:
    """Run WMH segmentation with uncertainty estimation."""
    from src.model import WMHModel, UNet3D
    import torchio as tio
    from src.datamodules.transforms import get_preprocessing

    device = _get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    net = UNet3D(mc_dropout_ratio=mc_ratio)
    model = WMHModel.load_from_checkpoint(checkpoint, net=net, map_location=device)
    model.eval()
    model.to(device)

    # Load and preprocess
    subject = tio.Subject(
        t1=tio.ScalarImage(str(t1_path)),
        flair=tio.ScalarImage(str(flair_path)),
    )
    transform = get_preprocessing(include_labels=False)
    subject = transform(subject)

    # Concatenate T1 + FLAIR
    t1_data = subject.t1.data
    flair_data = subject.flair.data
    input_tensor = torch.cat([t1_data, flair_data], dim=0).unsqueeze(0).to(device)

    # MC dropout inference
    predictions = []
    with torch.no_grad():
        for _ in range(mc_samples):
            model.train()  # enable dropout
            output = model.net(input_tensor)
            prob = torch.softmax(output, dim=1)[:, 1]  # WMH probability
            predictions.append(prob.cpu())

    preds = torch.stack(predictions)
    mean_pred = preds.mean(dim=0).squeeze().numpy()
    uncertainty = preds.std(dim=0).squeeze().numpy()
    hard_pred = (mean_pred > 0.5).astype(np.uint8)

    # Save outputs as NIfTI (preserve T1 geometry)
    ref_img = sitk.ReadImage(str(t1_path))
    outputs = {}

    for name, arr in [("wmh_mask", hard_pred), ("wmh_prob", mean_pred), ("wmh_uncertainty", uncertainty)]:
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.CopyInformation(ref_img)
        path = output_dir / f"{name}.nii.gz"
        sitk.WriteImage(img, str(path))
        outputs[name] = str(path)

    # Measurements
    wmh_voxels = int(hard_pred.sum())
    voxel_vol = np.prod(ref_img.GetSpacing())
    outputs["measurements"] = {
        "wmh_volume_mm3": round(wmh_voxels * voxel_vol, 1),
        "wmh_voxels": wmh_voxels,
        "mean_uncertainty": round(float(uncertainty[hard_pred > 0].mean()), 4) if wmh_voxels > 0 else 0,
        "mc_samples": mc_samples,
    }
    print(f"  WMH volume: {outputs['measurements']['wmh_volume_mm3']} mm³")

    return outputs


def _find_inputs(directory: Path) -> tuple[Path | None, Path | None]:
    """Find T1 and FLAIR volumes in input directory."""
    t1 = flair = None
    for f in directory.iterdir():
        name = f.name.lower()
        if "t1" in name and f.suffix in (".gz", ".nii"):
            t1 = f
        elif "flair" in name and f.suffix in (".gz", ".nii"):
            flair = f
    return t1, flair


def _write_result(out_dir: Path, status: str, outputs=None, error=None):
    result = {"status": status}
    if outputs:
        result["outputs"] = [
            {"type": k, "file": Path(v).name}
            for k, v in outputs.items() if isinstance(v, str) and Path(v).exists()
        ]
        if "measurements" in outputs:
            result["measurements"] = outputs["measurements"]
    if error:
        result["error"] = error
    vol = outputs.get("measurements", {}).get("wmh_volume_mm3", "?") if outputs else "?"
    result["__mercure_notification"] = {"text": f"WMH segmentation: {status}, volume: {vol} mm³"}
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    run()
