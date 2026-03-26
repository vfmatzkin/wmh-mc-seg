import marimo

app = marimo.App(width="medium")


@app.cell
def _imports():
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    import marimo as mo

    return mo, plt, nib, np


@app.cell
def _ui_controls(mo):
    subject_path = mo.ui.text(
        value="~/Code/datasets/wmh/training/Utrecht/0",
        label="Subject path",
        full_width=True,
    )
    run_name_ce = mo.ui.text(
        value="training_Utrecht_Amsterdam_Singapore_CE_3684_best",
        label="CE run name",
        full_width=True,
    )
    run_name_meep = mo.ui.text(
        value="training_Utrecht_Amsterdam_Singapore_MEEP_6996_best",
        label="CE+MEEP run name",
        full_width=True,
    )
    return subject_path, run_name_ce, run_name_meep


@app.cell
def _slice_controls(mo):
    slice_z = mo.ui.slider(start=0, stop=100, value=28, label="Axial slice (z)")
    slice_x = mo.ui.slider(start=0, stop=250, value=152, label="Sagittal slice (x)")
    show_entropy = mo.ui.checkbox(label="Show entropy maps (Fig 2)", value=False)
    return slice_z, slice_x, show_entropy


@app.cell
def _load_niftis(mo, nib, np, subject_path, run_name_ce, run_name_meep):
    import os

    _subj = os.path.expanduser(subject_path.value)
    _run_ce = run_name_ce.value
    _run_meep = run_name_meep.value

    _flair_path = os.path.join(_subj, "pre/FLAIR.nii.gz")
    _gt_path = os.path.join(_subj, "wmh.nii.gz")
    _pred_ce_path = os.path.join(_subj, f"pred_wmh_softmax_{_run_ce}.nii.gz")
    _pred_meep_path = os.path.join(_subj, f"pred_wmh_softmax_{_run_meep}.nii.gz")

    _missing = [p for p in [_flair_path, _pred_ce_path, _pred_meep_path] if not os.path.exists(p)]
    mo.stop(
        len(_missing) > 0,
        mo.callout(
            mo.md("**Files not found:**\n" + "\n".join(f"- `{p}`" for p in _missing)),
            kind="danger",
        ),
    )

    _flair_img = nib.load(_flair_path)
    flair = _flair_img.get_fdata()
    _header = _flair_img.header

    _vox = _header.get_zooms()[:3]
    _dx, _dy, _dz = _vox
    aspect_axial = float(_dy / _dx) if _dx > 0 else 1.0
    aspect_sagittal = float(_dz / _dy) if _dy > 0 else 1.0

    _ce_softmax = nib.load(_pred_ce_path).get_fdata()
    ce_prob = _ce_softmax[..., 1] if _ce_softmax.shape[-1] == 2 else _ce_softmax

    _meep_softmax = nib.load(_pred_meep_path).get_fdata()
    meep_prob = _meep_softmax[..., 1] if _meep_softmax.shape[-1] == 2 else _meep_softmax

    def _entropy_map(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    ce_entropy = _entropy_map(ce_prob)
    meep_entropy = _entropy_map(meep_prob)

    return flair, ce_prob, meep_prob, ce_entropy, meep_entropy, aspect_axial, aspect_sagittal


@app.cell
def _fig1(mo, plt, np, flair, ce_prob, meep_prob, aspect_axial, aspect_sagittal, slice_z, slice_x):
    FIGURE_FACECOLOR = "#060606"
    PROB_CMAP = "hot"
    SMALL_SIZE = 11
    MEDIUM_SIZE = 13

    _z = min(slice_z.value, flair.shape[2] - 1)
    _x = min(slice_x.value, flair.shape[0] - 1)

    _plot_keys = ["FLAIR", "CE_Prob", "CE_MEEP_Prob"]
    _titles = ["FLAIR", "CE Softmax", "CE+MEEP Softmax"]
    _imgs = {"FLAIR": flair, "CE_Prob": ce_prob, "CE_MEEP_Prob": meep_prob}

    _fig1, _axs = plt.subplots(2, 3, figsize=(11, 7.5))
    _fig1.set_facecolor(FIGURE_FACECOLOR)
    for _row_axes in _axs:
        for _a in _row_axes:
            _a.set_facecolor(FIGURE_FACECOLOR)

    _im_prob = None
    for _col, (_key, _title) in enumerate(zip(_plot_keys, _titles)):
        _ax_axial = _axs[0][_col]
        if _key == "FLAIR":
            _ax_axial.imshow(np.rot90(_imgs[_key][:, :, _z]), cmap="gray", aspect=aspect_axial)
        else:
            _ax_axial.imshow(np.rot90(flair[:, :, _z]), cmap="gray", aspect=aspect_axial)
            _im_prob = _ax_axial.imshow(
                np.rot90(_imgs[_key][:, :, _z]),
                cmap=PROB_CMAP,
                alpha=0.6,
                vmin=0,
                vmax=1,
                aspect=aspect_axial,
            )
        _ax_axial.set_title(_title, color="white", fontsize=MEDIUM_SIZE)
        _ax_axial.axis("off")

        _ax_sag = _axs[1][_col]
        if _key == "FLAIR":
            _ax_sag.imshow(np.rot90(_imgs[_key][_x, :, :]), cmap="gray", aspect=aspect_sagittal)
        else:
            _ax_sag.imshow(np.rot90(flair[_x, :, :]), cmap="gray", aspect=aspect_sagittal)
            _ax_sag.imshow(
                np.rot90(_imgs[_key][_x, :, :]),
                cmap=PROB_CMAP,
                alpha=0.6,
                vmin=0,
                vmax=1,
                aspect=aspect_sagittal,
            )
        _ax_sag.axis("off")

        if _col == 0:
            _axs[0][0].text(
                -0.07,
                0.5,
                "Axial",
                rotation=90,
                va="center",
                ha="right",
                transform=_axs[0][0].transAxes,
                color="white",
                fontsize=MEDIUM_SIZE,
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
                fontsize=MEDIUM_SIZE,
            )

    if _im_prob is not None:
        _cbar_ax = _fig1.add_axes([0.91, 0.3, 0.02, 0.4])
        _cbar = _fig1.colorbar(_im_prob, cax=_cbar_ax)
        _cbar.set_label("Softmax Probability", color="white", size=SMALL_SIZE)
        _cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        _cbar.outline.set_edgecolor("white")

    _fig1.suptitle("Fig 1 — FLAIR + Softmax predictions", color="white", fontsize=MEDIUM_SIZE)
    _fig1.tight_layout(rect=[0, 0, 0.9, 1])

    fig1_widget = mo.mpl.interactive(_fig1)
    return (fig1_widget,)


@app.cell
def _fig2(mo, plt, np, flair, ce_entropy, meep_entropy, aspect_axial, slice_z, show_entropy):
    mo.stop(
        not show_entropy.value,
        mo.md("Enable the **entropy checkbox** above to see entropy maps."),
    )

    import matplotlib.gridspec as gridspec

    FIGURE_FACECOLOR_2 = "#060606"
    ENTROPY_CMAP = "viridis"
    MEDIUM_SIZE_2 = 12

    _z2 = min(slice_z.value, flair.shape[2] - 1)

    _data_fig2 = {"FLAIR": flair, "CE_Entropy": ce_entropy, "CE_MEEP_Entropy": meep_entropy}
    _keys2 = ["FLAIR", "CE_Entropy", "CE_MEEP_Entropy"]
    _titles2 = ["FLAIR", "CE Entropy", "CE+MEEP Entropy"]

    _fig2 = plt.figure(figsize=(11.5, 5))
    _fig2.set_facecolor(FIGURE_FACECOLOR_2)

    _gs = gridspec.GridSpec(1, 4, figure=_fig2, width_ratios=[1, 1, 1, 0.07], wspace=0.1)

    _im_ent = None
    for _col2, (_key2, _title2) in enumerate(zip(_keys2, _titles2)):
        _ax2 = _fig2.add_subplot(_gs[0, _col2])
        _ax2.set_facecolor(FIGURE_FACECOLOR_2)
        _slice2 = np.rot90(_data_fig2[_key2][:, :, _z2])
        if _key2 == "FLAIR":
            _ax2.imshow(_slice2, cmap="gray", aspect=aspect_axial)
        else:
            _ax2.imshow(np.rot90(flair[:, :, _z2]), cmap="gray", aspect=aspect_axial)
            _im_ent = _ax2.imshow(
                _slice2,
                cmap=ENTROPY_CMAP,
                alpha=0.6,
                vmin=0,
                vmax=np.log2(2),
                aspect=aspect_axial,
            )
        _ax2.set_title(_title2, color="white", fontsize=MEDIUM_SIZE_2)
        _ax2.axis("off")

    if _im_ent is not None:
        _gs_cbar = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=_gs[0, 3], hspace=0.5)
        _cax_ent = _fig2.add_subplot(_gs_cbar[1, 0])
        _cbar_ent = _fig2.colorbar(_im_ent, cax=_cax_ent, orientation="vertical")
        _cbar_ent.set_label("Voxel Entropy", color="white", size=MEDIUM_SIZE_2, labelpad=10)
        _cbar_ent.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        _cbar_ent.outline.set_edgecolor("white")

    _fig2.suptitle("Fig 2 — Entropy maps", color="white", fontsize=MEDIUM_SIZE_2)

    fig2_widget = mo.mpl.interactive(_fig2)
    return (fig2_widget,)


@app.cell
def _layout(
    mo,
    subject_path,
    run_name_ce,
    run_name_meep,
    slice_z,
    slice_x,
    show_entropy,
    fig1_widget,
    fig2_widget,
):
    _controls = mo.vstack(
        [
            mo.md("## NIfTI Viewer — Fig 1 & 2"),
            mo.md("### Data paths"),
            subject_path,
            run_name_ce,
            run_name_meep,
            mo.md("### Slice controls"),
            slice_z,
            slice_x,
            show_entropy,
        ]
    )
    return mo.vstack([_controls, fig1_widget, fig2_widget])


if __name__ == "__main__":
    app.run()
