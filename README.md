# WMH MRI Segmentation

3D U-Net with Monte Carlo dropout for White Matter Hyperintensity segmentation from brain MRI (T1 + FLAIR). Trained on the [WMH Segmentation Challenge](https://wmh.isi.uu.nl/) dataset (Utrecht, Amsterdam, Singapore).

Includes MEEP, KL, and MEALL uncertainty calibration losses.

> [!NOTE]
> This repo is under active refactoring. The original research code (used for my thesis) is available as [`v0.1.0`](../../releases/tag/v0.1.0). The current version modernizes the codebase (Lightning 2.x, PyTorch 2.x, proper packaging) without changing the model architecture or training logic.

## Project Structure

```
src/
  models/
    unet3d.py           # 3D U-Net (MONAI)
    wmh_module.py        # Lightning training module
    inference.py         # MC dropout inference (pure functions)
  losses/
    composite.py         # RegularizedLoss (MEEP/KL/MEALL/MEOOD)
    regularizers.py      # Uncertainty regularization terms
  datamodules/
    WMHDataModule.py     # Data loading and patch sampling
    transforms.py        # Shared preprocessing pipeline
  utils/
    metrics.py           # Dice score computation
    sitk_io.py           # NIfTI I/O with metadata preservation
    cli.py               # MLproject defaults loader
  train.py              # Training entry point
  predict.py            # Inference entry point
  plot.py               # Analysis and visualization
tests/                  # Unit tests (losses, transforms, metrics, CLI)
```

## Setup

### Docker (recommended)

```bash
# Set your datasets path
export DATASETS_PATH=/path/to/your/datasets

docker-compose up
```

This starts a Jupyter server. Connect via the URL in the terminal output.

### Manual

```bash
python3.10 -m venv env-wmh-mc-seg
source env-wmh-mc-seg/bin/activate
pip install -e ".[dev]"
```

## Training

Default hyperparameters live in `MLproject`. Any CLI argument overrides them.

```bash
python src/train.py --centers='training:Singapore' --loss='KL'
```

Or via MLflow:

```bash
mlflow run . -P centers='training:Singapore' -P loss='KL' --env-manager=local
```

> [!TIP]
> Run `python src/train.py --help` for the full parameter list.

### Available losses

| Loss | CLI name | Description |
|------|----------|-------------|
| Cross Entropy | `ce` | Standard CE |
| Dice | `dice` | MONAI Dice loss |
| CE + MEEP | `meep`, `cemeep` | Max Entropy on Erroneous Predictions |
| CE + KL | `kl`, `cekl` | KL divergence regularization |
| CE + MEALL | `meall`, `cemeall` | Max Entropy on All predictions |
| Dice + MEEP | `dicemeep`, `dmeep` | Dice with MEEP regularization |
| Dice + KL | `dicekl` | Dice with KL regularization |
| Dice + MEALL | `dicemeall` | Dice with MEALL regularization |
| CE + MEOOD | `meood`, `cemeood` | Max Entropy on OOD data |

All regularized losses accept `--meep-lambda` (regularization weight, default 0.3) and `--reg-start` (epoch to start regularization).

### Default hyperparameters

| Parameter | Value |
|-----------|-------|
| epochs | 800 |
| batch size | 64 |
| learning rate | 0.001 |
| dropout | 0.2 |
| loss | DiceMEALL |
| patch size | 32 |
| samples per volume | 12 |
| seed | 42 |

## Inference

```bash
python src/predict.py \
  --centers='test:Singapore' \
  --model-path='checkpoints/training_Singapore_best.ckpt' \
  --mc-samples=10 \
  --mc-ratio=0.1
```

MC dropout runs multiple forward passes with dropout enabled, then computes mean prediction and per-voxel uncertainty (standard deviation across passes).

## Interactive Analysis

Two [Marimo](https://marimo.io/) apps replace the old Jupyter notebooks.

```bash
pip install ".[interactive]"

# NIfTI slice viewer (fig1 + fig2)
marimo run marimo/viewer.py

# Analysis dashboard with tabs for fig3-7
marimo run marimo/analysis.py
```

The analysis app supports CSV caching (toggle in sidebar) so heavy computation only runs once. Cached data goes to `data/cache/`.

> [!WARNING]
> The Marimo apps require prediction outputs from trained models. Run training and inference first.

## Tests

```bash
pytest
```

78 tests covering losses, transforms, metrics, CLI defaults, integration (synthetic data), and analysis cache behavior.

## Data

Expects the [WMH Challenge](https://wmh.isi.uu.nl/) dataset structured as:

```
data_root/
  training/
    Utrecht/
      0/
        pre/T1.nii.gz
        pre/FLAIR.nii.gz
        wmh.nii.gz
    Amsterdam/
      GE3T/0/...
      GE1T5/0/...
    Singapore/
      0/...
  test/
    ...
```

## Citation

If you use this code, please cite:

> Matzkin, F. et al. "Towards reliable WMH segmentation under domain shift: An application study using maximum entropy regularization to improve uncertainty estimation." *Computers in Biology and Medicine*, vol. 196, Part A, 110639, 2025. [DOI](https://doi.org/10.1016/j.compbiomed.2025.110639)

```bibtex
@article{matzkin2025wmh,
  title={Towards reliable {WMH} segmentation under domain shift: An application study using maximum entropy regularization to improve uncertainty estimation},
  author={Matzkin, Franco and Larrazabal, Agostina and Milone, Diego H. and Dolz, Jose and Ferrante, Enzo},
  journal={Computers in Biology and Medicine},
  volume={196},
  pages={110639},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2025.110639}
}
```

## Stack

Python 3.10+, PyTorch 2.x, Lightning 2.x, MONAI, TorchIO, MLflow.
