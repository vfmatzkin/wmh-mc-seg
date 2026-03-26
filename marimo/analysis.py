import marimo

app = marimo.App(width="full")


@app.cell
def _imports():
    import os
    import sys

    import marimo as mo

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.lines import Line2D
    from scipy.stats import pearsonr

    from analysis import (
        build_plot_data,
        compute_all_metrics,
        confusion_entropy_data,
        dice_vs_entropy_data,
        entropy_volume_ranges,
        reliability_data,
    )

    return (
        mo,
        sys,
        os,
        build_plot_data,
        dice_vs_entropy_data,
        confusion_entropy_data,
        compute_all_metrics,
        entropy_volume_ranges,
        reliability_data,
        plt,
        mpatches,
        Line2D,
        sns,
        pd,
        np,
        pearsonr,
    )


@app.cell
def _controls(mo):
    data_root = mo.ui.text(
        value="~/Code/datasets/wmh",
        label="Data root",
        full_width=True,
    )
    loss_select = mo.ui.multiselect(
        options=["CE", "CE_MEEP", "CE_KL", "CE_MEALL"],
        value=["CE", "CE_MEEP", "CE_KL", "CE_MEALL"],
        label="Losses to compare",
    )
    run_ce = mo.ui.text(
        value="training_Utrecht_Amsterdam_Singapore_CE_3684_best",
        label="CE run name",
        full_width=True,
    )
    run_meep = mo.ui.text(
        value="training_Utrecht_Amsterdam_Singapore_MEEP_6996_best",
        label="CE_MEEP run name",
        full_width=True,
    )
    run_kl = mo.ui.text(
        value="training_Utrecht_Amsterdam_Singapore_cekl_4187_best",
        label="CE_KL run name",
        full_width=True,
    )
    run_meall = mo.ui.text(
        value="training_Utrecht_Amsterdam_Singapore_MEALL_8766_best",
        label="CE_MEALL run name",
        full_width=True,
    )
    use_cache = mo.ui.checkbox(label="Use CSV cache", value=True)
    run_button = mo.ui.run_button(label="Compute")
    return data_root, loss_select, run_ce, run_meep, run_kl, run_meall, use_cache, run_button


@app.cell
def _compute(
    mo,
    os,
    build_plot_data,
    dice_vs_entropy_data,
    confusion_entropy_data,
    compute_all_metrics,
    entropy_volume_ranges,
    reliability_data,
    data_root,
    loss_select,
    run_ce,
    run_meep,
    run_kl,
    run_meall,
    use_cache,
    run_button,
):
    mo.stop(
        not run_button.value,
        mo.callout(
            mo.md("Configure settings and click **Compute** to generate figures."),
            kind="info",
        ),
    )

    _root = os.path.expanduser(data_root.value)
    _selected = set(loss_select.value)

    _runs = {
        "CE UtAmSi": run_ce.value,
        "CE_MEEP UtAmSi": run_meep.value,
        "CE_KL UtAmSi": run_kl.value,
        "CE_MEALL UtAmSi": run_meall.value,
    }
    _runs = {k: v for k, v in _runs.items() if k.split()[0] in _selected}

    _plot_data = build_plot_data(_root)
    _plot_data["runs_to_compare"] = _runs
    _plot_data["losses"] = [loss for loss in _plot_data["losses"] if loss in _selected]

    _cache_dir = "data/cache" if use_cache.value else None
    if _cache_dir:
        os.makedirs(_cache_dir, exist_ok=True)

    def _cp(name):
        return os.path.join(_cache_dir, name) if _cache_dir else None

    df_dice_ent = dice_vs_entropy_data(_plot_data, _runs, cache_path=_cp("fig3.csv"))
    df_confusion = confusion_entropy_data(_plot_data, _runs, cache_path=_cp("fig4.csv"))
    df_metrics = compute_all_metrics(_plot_data, _runs, cache_path=_cp("fig5.csv"))
    df_volume = entropy_volume_ranges(_plot_data, _runs, cache_path=_cp("fig6.csv"))
    df_reliability = reliability_data(_plot_data, _runs, cache_path=_cp("fig7.csv"))

    return df_dice_ent, df_confusion, df_metrics, df_volume, df_reliability


@app.cell
def _fig3(mo, plt, sns, np, pearsonr, df_dice_ent):
    SMALL_SIZE = 15
    MEDIUM_SIZE = 17

    _rename = {"UtAmSi": "In-distribution", "UMCL": "Out-of-distribution"}
    _df = df_dice_ent.copy()
    _df["Test_Center"] = _df["Test_Center"].map(lambda c: _rename.get(c, c))

    _unique_losses = _df["Loss"].unique()
    _unique_centers = _df["Test_Center"].unique()

    _fig3, _ax3 = plt.subplots(figsize=(9, 8))
    sns.set_style("whitegrid")

    sns.scatterplot(
        data=_df,
        x="Dice",
        y="Entropy",
        hue="Loss",
        style="Test_Center",
        s=80,
        ax=_ax3,
    )
    for _loss in _unique_losses:
        _sub = _df[_df["Loss"] == _loss]
        if not _sub.empty:
            sns.regplot(
                x="Dice",
                y="Entropy",
                data=_sub,
                scatter=False,
                ci=None,
                ax=_ax3,
                line_kws={"linewidth": 1.5},
            )

    _ax3.set_xlabel("Dice Coefficient", fontsize=MEDIUM_SIZE)
    _ax3.set_ylabel("Entropy", fontsize=MEDIUM_SIZE)
    _ax3.set_xlim([0, 1])
    _ax3.set_ylim([0, 1])
    _ax3.set_aspect("equal", adjustable="box")
    if _ax3.get_legend():
        _ax3.get_legend().remove()

    _palette = sns.color_palette("tab10", n_colors=len(_unique_losses))
    _loss_elems = [plt.Line2D([0], [0], marker="", linestyle="", label="Loss")]
    for _i, _loss in enumerate(_unique_losses):
        _sub = _df[_df["Loss"] == _loss]
        _corr = 0.0
        if len(_sub) > 1:
            try:
                _corr, _ = pearsonr(_sub["Dice"], _sub["Entropy"])
            except Exception:
                pass
        _lname = f"${_loss.split('_')[0]}_{{{_loss.split('_', 1)[1]}}}$" if "_" in _loss else _loss
        _loss_elems.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color=_palette[_i],
                linestyle="",
                markersize=12,
                label=f"{_lname} ({_corr:.3f})",
            )
        )

    _center_elems = [plt.Line2D([0], [0], marker="", linestyle="", label="Data")]
    for _center in sorted(_unique_centers):
        _marker = "o" if "Out" in _center else "x"
        _center_elems.append(
            plt.Line2D(
                [0], [0], marker=_marker, color="black", linestyle="", markersize=12, label=_center
            )
        )

    _max_len = max(len(_loss_elems), len(_center_elems))
    _loss_elems += [plt.Line2D([0], [0], marker="", linestyle="", label="")] * (
        _max_len - len(_loss_elems)
    )
    _center_elems += [plt.Line2D([0], [0], marker="", linestyle="", label="")] * (
        _max_len - len(_center_elems)
    )

    _ax3.legend(
        handles=_loss_elems + _center_elems,
        fontsize=SMALL_SIZE,
        loc="upper right",
        ncol=2,
        columnspacing=1,
    )
    plt.tight_layout()

    fig3_widget = mo.mpl.interactive(_fig3)
    return (fig3_widget,)


@app.cell
def _fig4(mo, plt, sns, np, pd, Line2D, df_confusion):
    SMALL_SIZE_4 = 15
    MEDIUM_SIZE_4 = 17
    BIGGER_SIZE_4 = 19

    _color_in = "#4878AA"
    _color_out = "#D28C5F"

    _df4 = df_confusion.copy()
    _losses_order = [
        loss for loss in ["CE", "CE_MEEP", "CE_KL", "CE_MEALL"] if loss in _df4["Loss"].unique()
    ]
    _cats = ["TP", "TN", "FP", "FN"]

    _fig4, _axes4 = plt.subplots(2, 2, figsize=(10, 11), sharey=True)
    _axes4 = _axes4.flatten()
    sns.set_style("white")

    _x_pos = np.arange(len(_losses_order) * 2)

    for _i, _cat in enumerate(_cats):
        _ax = _axes4[_i]
        _cat_df = _df4[_df4["Category"] == _cat]

        for _j, _dist in enumerate(["In-distribution", "Out-of-distribution"]):
            _color = _color_in if _dist == "In-distribution" else _color_out
            _dist_df = _cat_df[_cat_df["Distribution"] == _dist]
            for _k, _loss in enumerate(_losses_order):
                _vals = _dist_df[_dist_df["Loss"] == _loss]["Entropy"]
                _x = _x_pos[_k + _j * len(_losses_order)]
                if len(_vals) > 0:
                    _ax.scatter(
                        _x + np.random.normal(0, 0.05, len(_vals)),
                        _vals,
                        color=_color,
                        alpha=0.2,
                        s=10,
                    )
                    _ax.scatter(
                        _x,
                        float(np.median(_vals)),
                        marker="^",
                        color="black",
                        edgecolors="white",
                        linewidths=0.5,
                        s=100,
                        zorder=3,
                    )

        _ax.set_title(_cat, fontsize=BIGGER_SIZE_4, pad=10)
        if _i >= 2:
            _ax.set_xlabel("Loss", fontsize=MEDIUM_SIZE_4)
        if _i % 2 == 0:
            _ax.set_ylabel("Entropy", fontsize=MEDIUM_SIZE_4)
        _ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE_4)
        _ax.set_xticks(_x_pos)

        _xlabels = []
        for _lbl in _losses_order * 2:
            if "_" in _lbl:
                _p = _lbl.split("_", 1)
                _xlabels.append(rf"${_p[0]}_{{{_p[1]}}}$")
            else:
                _xlabels.append(rf"${_lbl}$")
        _ax.set_xticklabels(_xlabels, rotation=45, ha="right")
        _ax.grid(False)

    _legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            linestyle="None",
            markerfacecolor=_color_in,
            markersize=8,
            label="In-distribution",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            linestyle="None",
            markerfacecolor=_color_out,
            markersize=8,
            label="Out-of-distribution",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=10,
            label="Median",
        ),
    ]
    _axes4[1].legend(handles=_legend_elems, loc="upper right", fontsize=SMALL_SIZE_4)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    fig4_widget = mo.mpl.interactive(_fig4)
    return (fig4_widget,)


@app.cell
def _fig5(mo, plt, sns, np, df_metrics):
    SMALL_SIZE_5 = 10
    MEDIUM_SIZE_5 = 12
    BIGGER_SIZE_5 = 14

    try:
        from statannot import add_stat_annotation

        _has_statannot = True
    except ImportError:
        _has_statannot = False

    _df5 = df_metrics.copy()
    _df5 = _df5[_df5["Center"] == "UtAmSi"]
    _metrics = ("Entropy", "Dice Score", "Hausdorff Distance")
    _df5 = _df5.dropna(subset=list(_metrics))
    _df5 = _df5[~np.isinf(_df5["Hausdorff Distance"])]

    _df5_melt = _df5.melt(
        id_vars=["Loss", "Distribution"],
        value_vars=list(_metrics),
        var_name="Metric",
        value_name="Value",
    )

    _hue_order = ("In-distribution", "Out-of-distribution")
    _palette5 = ["#5975a4", "#cc8963"]

    _fig5, _axes5 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    sns.set_style("whitegrid")

    for _ax5, _metric in zip(_axes5, _metrics):
        _sub5 = _df5_melt[_df5_melt["Metric"] == _metric]
        sns.boxplot(
            x="Loss",
            y="Value",
            hue="Distribution",
            hue_order=_hue_order,
            data=_sub5,
            ax=_ax5,
            width=0.7,
            fliersize=3,
            palette=_palette5,
        )

        if _has_statannot:
            _pairs = [
                ((L, _hue_order[0]), (L, _hue_order[1]))
                for L in _sub5["Loss"].unique()
                if ((_sub5["Loss"] == L) & (_sub5["Distribution"] == _hue_order[0])).any()
                and ((_sub5["Loss"] == L) & (_sub5["Distribution"] == _hue_order[1])).any()
            ]
            if _pairs:
                try:
                    add_stat_annotation(
                        _ax5,
                        data=_sub5,
                        x="Loss",
                        y="Value",
                        hue="Distribution",
                        box_pairs=_pairs,
                        test="Mann-Whitney",
                        text_format="star",
                        loc="inside",
                        verbose=0,
                    )
                except Exception:
                    pass

        _xt5 = []
        for _lbl5 in _ax5.get_xticklabels():
            _txt5 = _lbl5.get_text()
            if "_" in _txt5:
                _nm5, _sb5 = _txt5.split("_", 1)
                _xt5.append(rf"${_nm5}_{{{_sb5}}}$")
            else:
                _xt5.append(rf"${_txt5}$")
        _ax5.set_xticklabels(_xt5, fontsize=SMALL_SIZE_5)
        _ax5.set_title(_metric, fontsize=BIGGER_SIZE_5)
        _ax5.set_xlabel("", fontsize=MEDIUM_SIZE_5)
        _ax5.set_ylabel(
            _metric if _metric != "Hausdorff Distance" else "Hausdorff (mm)",
            fontsize=MEDIUM_SIZE_5,
        )
        _ax5.tick_params(axis="both", labelsize=SMALL_SIZE_5)
        if _ax5.get_legend():
            _ax5.get_legend().remove()

    _h5, _l5 = _axes5[0].get_legend_handles_labels()
    _axes5[0].legend(
        _h5,
        _l5,
        title="Distribution",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=1,
        frameon=True,
        fontsize=SMALL_SIZE_5,
        title_fontsize=MEDIUM_SIZE_5,
    )
    plt.tight_layout()

    fig5_widget = mo.mpl.interactive(_fig5)
    return (fig5_widget,)


@app.cell
def _fig6(mo, plt, np, mpatches, df_volume):
    SMALL_SIZE_6 = 10
    MEDIUM_SIZE_6 = 12

    _loss_colors = {
        "CE": "#4C72B0",
        "CE_MEALL": "#C44E52",
        "CE_MEEP": "#DD8452",
        "CE_KL": "#55A868",
    }
    _loss_order = ["CE", "CE_MEALL", "CE_MEEP", "CE_KL"]

    _df6 = df_volume.copy()
    if _df6.empty:
        _fig6 = plt.figure(figsize=(6, 4))
        _ax_empty6 = _fig6.add_subplot(111)
        _ax_empty6.text(
            0.5, 0.5, "No data available", ha="center", va="center", transform=_ax_empty6.transAxes
        )
    else:
        _vol_labels = _df6["Volume Range"].unique()
        _box_w = 0.38

        _fig6, _axes6 = plt.subplots(
            1, len(_vol_labels), figsize=(4.5 * len(_vol_labels), 5), sharey=True
        )
        if len(_vol_labels) == 1:
            _axes6 = [_axes6]

        for _ax_idx6, _vr in enumerate(_vol_labels):
            _ax6 = _axes6[_ax_idx6]
            _vrdf = _df6[_df6["Volume Range"] == _vr]
            _mn, _mx = float("inf"), float("-inf")

            _filtered = [x for x in _loss_order if x in _df6["Loss"].unique()]
            for _i6, _loss6 in enumerate(_filtered):
                _base = float(_i6)

                _p_in = _base - _box_w / 2
                _e_in = _vrdf[(_vrdf["Loss"] == _loss6) & (_vrdf["Center"] == "In-distribution")][
                    "Entropy"
                ]
                if not _e_in.empty:
                    _mn = min(_mn, _p_in - _box_w / 2)
                    _mx = max(_mx, _p_in + _box_w / 2)
                    _ax6.boxplot(
                        _e_in,
                        positions=[_p_in],
                        widths=_box_w,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(
                            facecolor=_loss_colors.get(_loss6, "#ccc"),
                            edgecolor="black",
                            linewidth=1.2,
                        ),
                        whiskerprops=dict(color="black", linewidth=1.2),
                        capprops=dict(color="black", linewidth=1.2),
                        medianprops=dict(color="black", linewidth=1.2),
                    )

                _p_out = _base + _box_w / 2
                _e_out = _vrdf[
                    (_vrdf["Loss"] == _loss6) & (_vrdf["Center"] == "Out-of-distribution")
                ]["Entropy"]
                if not _e_out.empty:
                    _mn = min(_mn, _p_out - _box_w / 2)
                    _mx = max(_mx, _p_out + _box_w / 2)
                    _lc6 = _loss_colors.get(_loss6, "#ccc")
                    _ax6.boxplot(
                        _e_out,
                        positions=[_p_out],
                        widths=_box_w,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor="white", edgecolor=_lc6, linewidth=1.2),
                        whiskerprops=dict(color=_lc6, linewidth=1.2),
                        capprops=dict(color=_lc6, linewidth=1.2),
                        medianprops=dict(color="black", linewidth=1.2),
                    )

            _ax6.set_xticks([])
            _ax6.set_xticklabels([])
            _ax6.set_xlabel(_vr, fontsize=MEDIUM_SIZE_6 - 1)
            if _mn < float("inf"):
                _ax6.set_xlim(_mn - 0.1, _mx + 0.1)
            _ax6.yaxis.grid(True, linestyle="-", linewidth=0.5, color="lightgrey", alpha=0.7)
            _ax6.spines["top"].set_visible(False)
            _ax6.spines["right"].set_visible(False)
            if _ax_idx6 == 0:
                _ax6.set_ylabel("Entropy", fontsize=MEDIUM_SIZE_6)

        _loss_handles6 = [
            mpatches.Patch(facecolor=_loss_colors.get(_l, "#ccc"), edgecolor="black", label=_l)
            for _l in _loss_order
            if _l in _df6["Loss"].unique()
        ]
        _dist_handles6 = [
            mpatches.Patch(facecolor="dimgrey", edgecolor="black", label="In-distribution"),
            mpatches.Patch(facecolor="white", edgecolor="black", label="Out-of-distribution"),
        ]
        _axes6[0].legend(
            handles=_loss_handles6 + _dist_handles6, loc="best", fontsize=SMALL_SIZE_6, frameon=True
        )
        plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15, wspace=0.1)

    fig6_widget = mo.mpl.interactive(_fig6)
    return (fig6_widget,)


@app.cell
def _fig7(mo, plt, np, df_reliability):
    SMALL_SIZE_7 = 10
    MEDIUM_SIZE_7 = 12
    BIGGER_SIZE_7 = 14

    _loss_colors7 = {
        "CE": "#4C72B0",
        "CE_MEEP": "#DD8452",
        "CE_KL": "#55A868",
        "CE_MEALL": "#C44E52",
    }
    _rename7 = {"UtAmSi": "In-distribution", "UMCL": "Out-of-distribution"}

    _df7 = df_reliability.copy()
    if _df7.empty:
        _fig7 = plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        _centers7 = sorted(
            _df7["Center"].unique().tolist(),
            key=lambda c: 0 if "In" in _rename7.get(c, c) else 1,
        )
        _n_c7 = len(_centers7)
        _losses7 = _df7["Loss"].unique().tolist()

        _fig7, _axes7 = plt.subplots(1, _n_c7, figsize=(4.5 * _n_c7, 5), sharey=True, sharex=True)
        if _n_c7 == 1:
            _axes7 = [_axes7]

        for _ax7 in _axes7:
            _ax7.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

        for _li7, _loss7 in enumerate(_losses7):
            _color7 = _loss_colors7.get(_loss7, plt.cm.tab10(_li7 % 10))
            _loss_df7 = _df7[_df7["Loss"] == _loss7]

            for _i7, _center7 in enumerate(_centers7):
                _cdf7 = _loss_df7[_loss_df7["Center"] == _center7]
                if _cdf7.empty:
                    continue
                _pp7 = _cdf7["PredProb"].values
                _ep7 = _cdf7["EmpProb"].values
                _ece7 = _cdf7["ECE"].iloc[0]

                _parts7 = _loss7.split("_")
                _ll7 = (
                    _parts7[0] + "".join([f"_{{{p}}}" for p in _parts7[1:]])
                    if len(_parts7) > 1
                    else _loss7
                )
                _ll7 = f"${_ll7}$"
                _ece_str7 = f"{_ece7:.1e}" if not np.isnan(_ece7) else "NaN"

                _axes7[_i7].plot(
                    _pp7,
                    _ep7,
                    "o-",
                    color=_color7,
                    markersize=5,
                    label=f"{_ll7} (ECE: {_ece_str7})",
                )

        for _i7, _ax7 in enumerate(_axes7):
            _ax7.set_xlabel("Predicted Probability", fontsize=MEDIUM_SIZE_7)
            if _i7 == 0:
                _ax7.set_ylabel("Empirical Probability", fontsize=MEDIUM_SIZE_7)
            _ax7.set_ylim([-0.05, 1.05])
            _ax7.set_xlim([-0.05, 1.05])

            _h7, _l7 = _ax7.get_legend_handles_labels()
            try:
                _pc_idx = _l7.index("Perfectly Calibrated")
                _h7.insert(0, _h7.pop(_pc_idx))
                _l7.insert(0, _l7.pop(_pc_idx))
            except ValueError:
                pass
            _ax7.legend(_h7, _l7, fontsize=SMALL_SIZE_7, loc="lower right")
            _ax7.set_title(_rename7.get(_centers7[_i7], _centers7[_i7]), fontsize=BIGGER_SIZE_7)
            _ax7.set_aspect("equal", adjustable="box")
            _ax7.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig7_widget = mo.mpl.interactive(_fig7)
    return (fig7_widget,)


@app.cell
def _downloads(mo, df_dice_ent, df_confusion, df_metrics, df_volume, df_reliability):
    downloads = mo.hstack(
        [
            mo.download(
                df_dice_ent.to_csv(index=False).encode(),
                filename="fig3_dice_entropy.csv",
                label="Fig 3 CSV",
            ),
            mo.download(
                df_confusion.to_csv(index=False).encode(),
                filename="fig4_confusion_entropy.csv",
                label="Fig 4 CSV",
            ),
            mo.download(
                df_metrics.to_csv(index=False).encode(),
                filename="fig5_metrics.csv",
                label="Fig 5 CSV",
            ),
            mo.download(
                df_volume.to_csv(index=False).encode(),
                filename="fig6_volume.csv",
                label="Fig 6 CSV",
            ),
            mo.download(
                df_reliability.to_csv(index=False).encode(),
                filename="fig7_reliability.csv",
                label="Fig 7 CSV",
            ),
        ]
    )
    return (downloads,)


@app.cell
def _tabs(mo, fig3_widget, fig4_widget, fig5_widget, fig6_widget, fig7_widget, downloads):
    _figures_tabs = mo.ui.tabs(
        {
            "Dice vs Entropy": fig3_widget,
            "Confusion Entropy": fig4_widget,
            "Metrics Comparison": fig5_widget,
            "Volume Ranges": fig6_widget,
            "Reliability": fig7_widget,
        }
    )
    tabs_panel = mo.vstack([_figures_tabs, mo.md("### Download data"), downloads])
    return (tabs_panel,)


@app.cell
def _sidebar(
    mo,
    data_root,
    loss_select,
    run_ce,
    run_meep,
    run_kl,
    run_meall,
    use_cache,
    run_button,
    tabs_panel,
):
    _sidebar = mo.vstack(
        [
            mo.md("## WMH Analysis Dashboard"),
            mo.md("### Settings"),
            data_root,
            loss_select,
            mo.md("**Run names**"),
            run_ce,
            run_meep,
            run_kl,
            run_meall,
            use_cache,
            run_button,
        ]
    )
    return mo.hstack([_sidebar, tabs_panel], widths=[1, 3], gap="1rem")


if __name__ == "__main__":
    app.run()
