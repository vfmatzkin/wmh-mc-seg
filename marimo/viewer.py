import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    import marimo as mo

    return mo, plt, nib, np


@app.cell
def _(mo):
    subject_path = mo.ui.text(
        value="~/Code/datasets/wmh/training/Utrecht/0",
        label="Subject path",
        full_width=True,
    )
    run_name_1 = mo.ui.text(
        value="training_Utrecht_meep_3932_best",
        label="Run 1 name",
        full_width=True,
    )
    run_name_2 = mo.ui.text(
        value="training_Utrecht_meep_3932_best",
        label="Run 2 name",
        full_width=True,
    )
    slice_z = mo.ui.slider(start=0, stop=100, value=28, label="Axial slice (z)")
    slice_x = mo.ui.slider(start=0, stop=250, value=152, label="Sagittal slice (x)")
    show_entropy = mo.ui.checkbox(label="Show entropy maps", value=False)

    mo.vstack(
        [
            mo.md("## NIfTI Viewer"),
            subject_path,
            mo.hstack([run_name_1, run_name_2]),
            mo.hstack([slice_z, slice_x, show_entropy]),
        ]
    )
    return subject_path, run_name_1, run_name_2, slice_z, slice_x, show_entropy


@app.cell
def _(mo, nib, np, subject_path, run_name_1, run_name_2):
    import os

    _subj = os.path.expanduser(subject_path.value)
    _run1 = run_name_1.value
    _run2 = run_name_2.value

    _flair_path = os.path.join(_subj, "pre/FLAIR.nii.gz")
    _pred1_path = os.path.join(_subj, f"pred_wmh_softmax_{_run1}.nii.gz")
    _pred2_path = os.path.join(_subj, f"pred_wmh_softmax_{_run2}.nii.gz")

    _missing = [p for p in [_flair_path, _pred1_path, _pred2_path] if not os.path.exists(p)]
    mo.stop(
        len(_missing) > 0,
        mo.callout(
            mo.md("**Files not found:**\n" + "\n".join(f"- `{p}`" for p in _missing)),
            kind="danger",
        ),
    )

    _flair_img = nib.load(_flair_path)
    flair = _flair_img.get_fdata()
    _vox = _flair_img.header.get_zooms()[:3]
    aspect_ax = float(_vox[1] / _vox[0]) if _vox[0] > 0 else 1.0
    aspect_sag = float(_vox[2] / _vox[1]) if _vox[1] > 0 else 1.0

    _s1 = nib.load(_pred1_path).get_fdata()
    prob1 = _s1[..., 1] if _s1.shape[-1] == 2 else _s1

    _s2 = nib.load(_pred2_path).get_fdata()
    prob2 = _s2[..., 1] if _s2.shape[-1] == 2 else _s2

    def _entropy(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    ent1 = _entropy(prob1)
    ent2 = _entropy(prob2)
    return flair, prob1, prob2, ent1, ent2, aspect_ax, aspect_sag


@app.cell
def _(
    mo,
    plt,
    np,
    flair,
    prob1,
    prob2,
    aspect_ax,
    aspect_sag,
    slice_z,
    slice_x,
    run_name_1,
    run_name_2,
):
    _bg = "#060606"
    _z = min(slice_z.value, flair.shape[2] - 1)
    _x = min(slice_x.value, flair.shape[0] - 1)

    _labels = ["FLAIR", run_name_1.value.split("_")[-2], run_name_2.value.split("_")[-2]]
    _data = [flair, prob1, prob2]

    _fig, _axs = plt.subplots(2, 3, figsize=(12, 8))
    _fig.set_facecolor(_bg)

    for _col in range(3):
        for _row, (_sl, _asp) in enumerate(
            [
                (np.rot90(flair[:, :, _z]), aspect_ax),
                (np.rot90(flair[_x, :, :]), aspect_sag),
            ]
        ):
            _a = _axs[_row][_col]
            _a.set_facecolor(_bg)
            _a.imshow(_sl, cmap="gray", aspect=_asp)
            if _col > 0:
                _ov = (
                    np.rot90(_data[_col][:, :, _z])
                    if _row == 0
                    else np.rot90(_data[_col][_x, :, :])
                )
                _a.imshow(_ov, cmap="hot", alpha=0.6, vmin=0, vmax=1, aspect=_asp)
            if _row == 0:
                _a.set_title(_labels[_col], color="white", fontsize=13)
            _a.axis("off")

    _axs[0][0].text(
        -0.07,
        0.5,
        "Axial",
        rotation=90,
        va="center",
        ha="right",
        transform=_axs[0][0].transAxes,
        color="white",
        fontsize=13,
    )
    _axs[1][0].text(
        -0.07,
        0.5,
        "Sagittal",
        rotation=90,
        va="center",
        ha="right",
        transform=_axs[1][0].transAxes,
        color="white",
        fontsize=13,
    )

    _fig.suptitle("FLAIR + Softmax Predictions", color="white", fontsize=14)
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo, plt, np, flair, ent1, ent2, aspect_ax, slice_z, show_entropy, run_name_1, run_name_2):
    mo.stop(
        not show_entropy.value,
        mo.md("*Enable the entropy checkbox above to see entropy maps.*"),
    )

    _bg = "#060606"
    _z = min(slice_z.value, flair.shape[2] - 1)
    _n1 = run_name_1.value.split("_")[-2]
    _n2 = run_name_2.value.split("_")[-2]
    _labels = ["FLAIR", f"{_n1} Entropy", f"{_n2} Entropy"]
    _data = [flair, ent1, ent2]

    _fig, _axs = plt.subplots(1, 3, figsize=(12, 4))
    _fig.set_facecolor(_bg)

    for _col in range(3):
        _a = _axs[_col]
        _a.set_facecolor(_bg)
        _sl = np.rot90(_data[_col][:, :, _z])
        if _col == 0:
            _a.imshow(_sl, cmap="gray", aspect=aspect_ax)
        else:
            _a.imshow(np.rot90(flair[:, :, _z]), cmap="gray", aspect=aspect_ax)
            _a.imshow(_sl, cmap="viridis", alpha=0.6, vmin=0, vmax=1, aspect=aspect_ax)
        _a.set_title(_labels[_col], color="white", fontsize=12)
        _a.axis("off")

    _fig.suptitle("Entropy Maps", color="white", fontsize=13)
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
