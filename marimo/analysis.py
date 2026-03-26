import marimo

app = marimo.App(width="full")


@app.cell
def _():
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
def _(mo):
    _wip = mo.callout(
        mo.md(
            "**WIP**: This dashboard uses placeholder run names. "
            "Replace with actual trained model runs when available."
        ),
        kind="warn",
    )
    _placeholder = "training_Utrecht_meep_3932_best"
    data_root = mo.ui.text(value="~/Code/datasets/wmh", label="Data root", full_width=True)
    loss_select = mo.ui.multiselect(
        options=["CE", "CE_MEEP", "CE_KL", "CE_MEALL"],
        value=["CE_MEEP"],
        label="Losses to compare",
    )
    run_ce = mo.ui.text(value=_placeholder, label="CE run", full_width=True)
    run_meep = mo.ui.text(value=_placeholder, label="CE_MEEP run", full_width=True)
    run_kl = mo.ui.text(value=_placeholder, label="CE_KL run", full_width=True)
    run_meall = mo.ui.text(value=_placeholder, label="CE_MEALL run", full_width=True)
    use_cache = mo.ui.checkbox(label="Use CSV cache", value=True)
    run_button = mo.ui.run_button(label="Compute")

    _controls = mo.vstack(
        [
            mo.md("## WMH Analysis Dashboard"),
            _wip,
            data_root,
            loss_select,
            mo.hstack([run_ce, run_meep]),
            mo.hstack([run_kl, run_meall]),
            mo.hstack([use_cache, run_button]),
        ]
    )
    _controls
    return (
        data_root,
        loss_select,
        run_ce,
        run_meep,
        run_kl,
        run_meall,
        use_cache,
        run_button,
    )


@app.cell
def _(
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
            mo.md("Configure settings above and click **Compute** to generate figures."),
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
    _plot_data["losses"] = [loss for loss in _plot_data.get("losses", []) if loss in _selected]

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

    mo.md("**Computation complete.** See figures below.")
    return df_dice_ent, df_confusion, df_metrics, df_volume, df_reliability


@app.cell
def _(mo, plt, sns, pearsonr, df_dice_ent):
    _df = df_dice_ent.copy()
    _rename = {"UtAmSi": "In-distribution", "UMCL": "Out-of-distribution"}
    _df["Test_Center"] = _df["Test_Center"].map(lambda c: _rename.get(c, c))

    _fig, _ax = plt.subplots(figsize=(9, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(
        data=_df,
        x="Dice",
        y="Entropy",
        hue="Loss",
        style="Test_Center",
        s=80,
        ax=_ax,
    )
    _ax.set_xlabel("Dice Coefficient", fontsize=15)
    _ax.set_ylabel("Entropy", fontsize=15)
    _ax.set_xlim([0, 1])
    _ax.set_ylim([0, 1])
    _ax.set_aspect("equal")
    plt.tight_layout()

    mo.vstack([mo.md("### Fig 3: Dice vs Entropy"), mo.mpl.interactive(_fig)])
    return


@app.cell
def _(mo, plt, sns, np, Line2D, df_confusion):
    _df = df_confusion.copy()
    _cats = ["TP", "TN", "FP", "FN"]
    _losses = list(_df["Loss"].unique())

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
    _axes = _axes.flatten()
    sns.set_style("white")

    for _i, _cat in enumerate(_cats):
        _a = _axes[_i]
        _cdf = _df[_df["Category"] == _cat]
        for _j, _loss in enumerate(_losses):
            _vals = _cdf[_cdf["Loss"] == _loss]["Entropy"]
            if len(_vals) > 0:
                _a.scatter(
                    _j + np.random.normal(0, 0.05, len(_vals)),
                    _vals,
                    alpha=0.2,
                    s=10,
                )
                _a.scatter(
                    _j,
                    float(np.median(_vals)),
                    marker="^",
                    color="black",
                    s=100,
                    zorder=3,
                )
        _a.set_title(_cat, fontsize=16)
        _a.set_xticks(range(len(_losses)))
        _a.set_xticklabels(_losses, rotation=45, ha="right")
        if _i % 2 == 0:
            _a.set_ylabel("Entropy")

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Fig 4: Entropy by Confusion Category"),
            mo.mpl.interactive(_fig),
        ]
    )
    return


@app.cell
def _(mo, plt, sns, np, df_metrics):
    _df = df_metrics.copy()
    _metrics = ["Entropy", "Dice Score", "Hausdorff Distance"]
    _df = _df.dropna(subset=[m for m in _metrics if m in _df.columns])

    _fig, _axes = plt.subplots(1, len(_metrics), figsize=(5 * len(_metrics), 5))
    if len(_metrics) == 1:
        _axes = [_axes]
    sns.set_style("whitegrid")

    for _a, _m in zip(_axes, _metrics):
        if _m in _df.columns:
            _sub = _df[~np.isinf(_df[_m])] if _m == "Hausdorff Distance" else _df
            sns.boxplot(x="Loss", y=_m, data=_sub, ax=_a)
            _a.set_title(_m)
            _a.set_xlabel("")

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### Fig 5: Metrics Comparison"),
            mo.mpl.interactive(_fig),
        ]
    )
    return


@app.cell
def _(mo, plt, df_volume):
    _df = df_volume.copy()
    if _df.empty:
        mo.callout(mo.md("No volume range data available."), kind="warn")
    else:
        _vols = _df["Volume Range"].unique()
        _fig, _axes = plt.subplots(
            1,
            max(1, len(_vols)),
            figsize=(4 * max(1, len(_vols)), 5),
            sharey=True,
        )
        if len(_vols) == 1:
            _axes = [_axes]
        for _i, _vr in enumerate(_vols):
            _sub = _df[_df["Volume Range"] == _vr]
            _axes[_i].boxplot(
                [_sub[_sub["Loss"] == _l]["Entropy"] for _l in _sub["Loss"].unique()],
                labels=_sub["Loss"].unique(),
            )
            _axes[_i].set_xlabel(_vr)
            if _i == 0:
                _axes[_i].set_ylabel("Entropy")
        plt.tight_layout()

        mo.vstack(
            [
                mo.md("### Fig 6: Entropy by Lesion Volume Range"),
                mo.mpl.interactive(_fig),
            ]
        )
    return


@app.cell
def _(mo, plt, np, df_reliability):
    _df = df_reliability.copy()
    if _df.empty:
        mo.callout(mo.md("No reliability data available."), kind="warn")
    else:
        _centers = _df["Center"].unique()
        _losses = _df["Loss"].unique()
        _fig, _axes = plt.subplots(
            1,
            max(1, len(_centers)),
            figsize=(5 * max(1, len(_centers)), 5),
            sharey=True,
        )
        if len(_centers) == 1:
            _axes = [_axes]

        for _i, _c in enumerate(_centers):
            _axes[_i].plot([0, 1], [0, 1], "k:", label="Perfect")
            for _loss in _losses:
                _sub = _df[(_df["Center"] == _c) & (_df["Loss"] == _loss)]
                if not _sub.empty:
                    _ece = _sub["ECE"].iloc[0]
                    _lbl = f"{_loss} (ECE: {_ece:.3f})" if not np.isnan(_ece) else _loss
                    _axes[_i].plot(
                        _sub["PredProb"],
                        _sub["EmpProb"],
                        "o-",
                        markersize=5,
                        label=_lbl,
                    )
            _axes[_i].set_title(_c)
            _axes[_i].set_xlabel("Predicted Probability")
            if _i == 0:
                _axes[_i].set_ylabel("Empirical Probability")
            _axes[_i].legend(fontsize=8)
            _axes[_i].set_aspect("equal")
            _axes[_i].set_xlim([-0.05, 1.05])
            _axes[_i].set_ylim([-0.05, 1.05])
        plt.tight_layout()

        mo.vstack(
            [
                mo.md("### Fig 7: Reliability Diagrams"),
                mo.mpl.interactive(_fig),
            ]
        )
    return


if __name__ == "__main__":
    app.run()
