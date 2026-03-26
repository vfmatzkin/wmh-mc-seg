from __future__ import annotations

import os.path
import subprocess

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
from calibration import get_ece
from medpy.metric.binary import dc as dice_score
from sklearn.calibration import calibration_curve

sns.set_style("whitegrid")

DEFAULT_COLORS = ["#3498db", "#2ecc71", "#e74c3c"]  # colors for each center
DEFAULT_CENTERS = ["utrecht", "singapore", "amsterdam"]

# DEFAULT_CSV_COLS takes the filenames from above before _training:
DEFAULT_CSV_COLS = [
    "pred_wmh_hard",
    "pred_wmh_softmax",
    "pred_logits",
    "gt_wmh",
    "pred_mc_logitsmean",
    "pred_mc_softmaxmean",
    "pred_mc_hardmean",
    "pred_mc_uncertmc",
]


def _load_center_csv(
    path_csvs: str,
    center: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if columns is None:
        columns = DEFAULT_CSV_COLS
    center_path = os.path.join(os.path.abspath(path_csvs), f"{center}.csv")
    df = pd.read_csv(center_path, header=None)
    df.columns = columns
    return df


def get_array_from_nifti(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def compare_runs(runs_to_compare: dict[str, str], metric_fn, **kwargs) -> None:
    for run_name, run_path in runs_to_compare.items():
        metric_fn(run_path, run_name, **kwargs)
        print("\n")


def dice_scores(
    path_csvs: str | None = None,
    alt_title: str = "",
    centers: list[str] | None = None,
    colors: list[str] | None = None,
) -> None:
    centers = centers or DEFAULT_CENTERS
    colors = colors or DEFAULT_COLORS

    dc = {}
    for center in centers:
        df = _load_center_csv(path_csvs, center)
        dc[center] = {"hard": []}

        for _, row in df.iterrows():
            pred_hard_path, pred_softmax_path, logits_path, gt_path, _, _, _, _ = row

            pred_hard = nib.load(pred_hard_path).get_fdata()
            gt = nib.load(gt_path).get_fdata()

            # Compute dice scores
            dice_hard = dice_score(pred_hard, gt)

            # Save results in dictionary
            dc[center]["hard"].append(dice_hard)

    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(5, 9), sharey=True)
    for i, center in enumerate(centers):
        sns.boxplot(y=dc[center]["hard"], color=colors[i], ax=axs[i], showmeans=True)
        axs[i].set_title(f"{center}")
        axs[i].set_ylim([0, 1])

    fig.suptitle((f"Dice Scores ({alt_title})" if alt_title else "Dice Scores"))
    fig.text(0.5, 0.04, "Center", ha="center")
    fig.text(0.04, 0.5, "Dice Score", va="center", rotation="vertical")

    fig.subplots_adjust(top=0.85, wspace=0.05)


def entropy(probs: np.ndarray, eps: float = 1e-3, apply_mean: bool = True) -> np.ndarray | float:
    probs = np.clip(probs, eps, 1 - eps)
    if apply_mean:
        return np.mean(-probs * np.log(probs) - (1 - probs) * np.log(1 - probs))
    else:
        return -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)


def get_b_mask_path(subj_path: str) -> str:
    """Load brain mask

    Given a subject path, check if the brain mask exists. If it does, return its
    path. If it doesn't, create it (using FSLs BET) and return its path.

    :param subj_path: Path to subject folder
    :return: Brain mask path
    """
    b_mask_path = os.path.join(subj_path, "pre", "T1_brain_mask.nii.gz")
    if os.path.exists(b_mask_path):
        return b_mask_path
    else:
        t1_path = os.path.join(subj_path, "pre", "T1.nii.gz")
        b_t1b_path = os.path.join(subj_path, "pre", "T1_brain.nii.gz")
        cmd = ["bet", t1_path, b_t1b_path, "-m"]
        subprocess.run(cmd, check=True)
        print(f"Created brain mask for {subj_path}")
        return b_mask_path


def entropy_segment_per_center(
    path_csvs: str | None = None,
    alt_title: str | None = None,
    centers: list[str] | None = None,
) -> None:
    centers = centers or DEFAULT_CENTERS

    segment_types = ["pred", "gt", "bMask"]
    ent = {center_type: {} for center_type in segment_types}

    for center in centers:
        df = _load_center_csv(path_csvs, center)

        for segment_type in segment_types:
            ent[segment_type][center] = {}

        for _, row in df.iterrows():
            pred_softmax_path, gt_path = row[1], row[3]
            subj_path = os.path.dirname(pred_softmax_path)
            subj = os.path.basename(subj_path)

            pred_softmax = nib.load(pred_softmax_path).get_fdata()
            gt = nib.load(gt_path).get_fdata()
            b_mask = nib.load(get_b_mask_path(subj_path)).get_fdata()

            pos_class = pred_softmax[:, :, :, 1].flatten()
            gt_class = gt.flatten()
            b_mask = b_mask.flatten()

            thres_pos = np.where(pos_class >= 0.5)[0]
            thres_gt_1 = np.where(gt_class == 1)[0]
            thres_brain = np.where(b_mask == 1)[0]

            filt_softmax = pos_class[thres_pos]
            filt_gt_1 = pos_class[thres_gt_1]
            filt_brain = pos_class[thres_brain]

            entropy_pred = entropy(filt_softmax)
            entropy_gt_1 = entropy(filt_gt_1)
            entropy_brain = entropy(filt_brain)

            ent["pred"][center][subj] = entropy_pred
            ent["gt"][center][subj] = entropy_gt_1
            ent["bMask"][center][subj] = entropy_brain

    # # Print the entropy for each center and individual
    # for segment_type in segment_types:
    #     print(f"Segment type: {segment_type}")
    #     for key, value in ent[segment_type].items():
    #         print(f"{key}: {value}")

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    for i, segment_type in enumerate(segment_types):
        for j, center in enumerate(centers):
            values = list(ent[segment_type][center].values())
            axs[i].boxplot(values, positions=[j], widths=0.6)

        # axs[i].set_xticks(range(len(centers)))
        axs[i].set_xticklabels(centers)
        axs[i].set_title(f"{segment_type}")

    if alt_title:
        plt.suptitle(alt_title)
    plt.show()


def append_round(imgs: np.ndarray, img, decimals: int = 2) -> np.ndarray:
    return np.concatenate((imgs, np.around(np.array(img), decimals)))


def probs_hist(
    path_csvs: str | None = None,
    alt_title: str | None = None,
    centers: list[str] | None = None,
) -> None:
    centers = centers or DEFAULT_CENTERS

    for center in centers:
        imgs_smx_0_0, imgs_smx_0_1, imgs_smx_1_0, imgs_smx_1_1 = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

        df = _load_center_csv(path_csvs, center)

        for _, row in df.iterrows():
            _, pred_wmh_softmax, _, gt_wmh, _, _, _, _ = row
            print(f"Processing {pred_wmh_softmax}...")

            pred_softmax = nib.load(pred_wmh_softmax).get_fdata()
            gt = nib.load(gt_wmh).get_fdata().flatten()

            gt_0 = np.where(gt == 0)[0]
            gt_1 = np.where(gt == 1)[0]

            # get the softmax vals for the voxels where gt is 1 in each channel
            img_smx_0_0 = pred_softmax[:, :, :, 0].flatten()[gt_0]
            img_smx_1_0 = pred_softmax[:, :, :, 1].flatten()[gt_0]
            img_smx_0_1 = pred_softmax[:, :, :, 0].flatten()[gt_1]
            img_smx_1_1 = pred_softmax[:, :, :, 1].flatten()[gt_1]

            # Concatenate sampled voxels for each class
            imgs_smx_0_0 = append_round(imgs_smx_0_0, img_smx_0_0)
            imgs_smx_1_0 = append_round(imgs_smx_1_0, img_smx_1_0)
            imgs_smx_0_1 = append_round(imgs_smx_0_1, img_smx_0_1)
            imgs_smx_1_1 = append_round(imgs_smx_1_1, img_smx_1_1)

        # Plot histogram for each voxel class
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        axs[0, 0].hist(imgs_smx_0_0, bins=20)
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_title(f"{center} - Class 0 - GT 0")
        axs[0, 1].hist(imgs_smx_1_0, bins=20)
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_title(f"{center} - Class 1 - GT 0")
        axs[1, 0].hist(imgs_smx_0_1, bins=20)
        axs[1, 0].set_yscale("log")
        axs[1, 0].set_title(f"{center} - Class 0 - GT 1")
        axs[1, 1].hist(imgs_smx_1_1, bins=20)
        axs[1, 1].set_yscale("log")
        axs[1, 1].set_title(f"{center} - Class 1 - GT 1")

        if alt_title:
            fig.suptitle(alt_title)

    plt.show()


def logits_hist(
    path_csvs: str | None = None,
    alt_title: str | None = None,
    centers: list[str] | None = None,
) -> None:
    centers = centers or DEFAULT_CENTERS

    for center in centers:
        imgs_logits_0_0, imgs_logits_0_1, imgs_logits_1_0, imgs_logits_1_1 = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

        df = _load_center_csv(path_csvs, center)

        for _, row in df.iterrows():
            _, _, pred_logits, gt_wmh, _, _, _, _ = row
            print(f"Processing {pred_logits}...")

            pred_logits = nib.load(pred_logits).get_fdata()
            gt = nib.load(gt_wmh).get_fdata().flatten()

            gt_0 = np.where(gt == 0)[0]
            gt_1 = np.where(gt == 1)[0]

            img_logits_0_0 = pred_logits[:, :, :, 0].flatten()[gt_0]
            img_logits_1_0 = pred_logits[:, :, :, 1].flatten()[gt_0]
            img_logits_0_1 = pred_logits[:, :, :, 0].flatten()[gt_1]
            img_logits_1_1 = pred_logits[:, :, :, 1].flatten()[gt_1]

            imgs_logits_0_0 = append_round(imgs_logits_0_0, img_logits_0_0)
            imgs_logits_1_0 = append_round(imgs_logits_1_0, img_logits_1_0)
            imgs_logits_0_1 = append_round(imgs_logits_0_1, img_logits_0_1)
            imgs_logits_1_1 = append_round(imgs_logits_1_1, img_logits_1_1)

        # Plot histogram for each voxel class
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        axs[0, 0].hist(imgs_logits_0_0, bins=200)
        # axs[0, 0].set_xlim([0, max_val])
        axs[0, 0].set_title(f"{center} - Class 0 - GT 0")

        axs[0, 1].hist(imgs_logits_1_0, bins=200)
        # axs[0, 1].set_xlim([0, max_val])
        axs[0, 1].set_title(f"{center} - Class 1 - GT 0")

        axs[1, 0].hist(imgs_logits_0_1, bins=200)
        # axs[1, 0].set_xlim([0, max_val])
        axs[1, 0].set_title(f"{center} - Class 0 - GT 1")

        axs[1, 1].hist(imgs_logits_1_1, bins=200)
        # axs[1, 1].set_xlim([0, max_val])
        axs[1, 1].set_title(f"{center} - Class 1 - GT 1")

        if alt_title:
            fig.suptitle(alt_title)

    plt.show()


def ece_reliability(
    path_csvs: str | None = None,
    alt_title: str | None = None,
    centers: list[str] | None = None,
) -> None:
    centers = centers or DEFAULT_CENTERS

    for center in centers:
        df = _load_center_csv(path_csvs, center)

        preds_1 = np.array([])
        gt_1 = np.array([], dtype=np.uint8)

        for i, row in df.iterrows():
            _, pred_wmh_softmax, _, gt_wmh, _, _, _, _ = row

            pred_softmax = get_array_from_nifti(pred_wmh_softmax)
            gt_wmh = get_array_from_nifti(gt_wmh)

            # One-hot encode the ground truth labels
            gt_one_hot = np.eye(2, dtype=np.uint8)[gt_wmh.astype(int)]

            # Flatten the arrays and append to numpy array
            preds_1 = np.concatenate((preds_1, pred_softmax[1].flatten()))
            gt_1 = np.concatenate((gt_1, gt_one_hot[:, :, :, 1].flatten()))

        # Define the number of bins for the reliability diagram
        num_bins = 10

        emp_probs_1, pred_probs_1 = calibration_curve(gt_1, preds_1, n_bins=num_bins)

        ece_0 = get_ece(preds_1, gt_1)

        # Plot the reliability diagram
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.plot(pred_probs_1, emp_probs_1, "o-", label="Foreground")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical probability")
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        ax.set_title(f"{center} - Reliability diagram\nECE: {ece_0:.4f}")

        if alt_title:
            fig.suptitle(alt_title)
    plt.show()


def dice_vs_entropy(
    path_csvs: str | None = None,
    alt_title: str | None = None,
    mask: str = "brain",
    centers: list[str] | None = None,
) -> None:
    """Dice vs entropy plot

    Compute for each subject the dice score and the entropy of the softmax, and
    place them in a scatter plot (dice in the y-axis and entropy in the x-axis).

    Each center will be colored differently.

    The y axis limits are 0-1 and the x axis its 0 to 0.2 more than the maximum

    """
    centers = centers or DEFAULT_CENTERS

    plt.figure()
    for center in centers:
        df = _load_center_csv(path_csvs, center)

        dice_ent = {center: []}

        for i, row in df.iterrows():
            pred_hard_path, pred_softmax_path, _, gt_path, _, _, _, _ = row
            subj_path = os.path.dirname(pred_softmax_path)

            pred_hard = nib.load(pred_hard_path).get_fdata()
            gt = nib.load(gt_path).get_fdata()

            pred_softmax = nib.load(pred_softmax_path).get_fdata()
            pos_class = pred_softmax[:, :, :, 1].flatten()

            # Load the mask
            if mask == "brain":
                p_mask = nib.load(get_b_mask_path(subj_path)).get_fdata()
                p_mask = p_mask.flatten()
            elif mask == "gt":
                p_mask = gt.flatten()
            elif mask == "softmax":
                p_mask = np.where(pos_class > 0.5, 1, 0)

            # Compute dice scores
            dice_hard = dice_score(pred_hard, gt)

            # Compute entropy
            thres_mask = np.where(p_mask == 1)[0]
            filt_softmax = pos_class[thres_mask]

            ent = entropy(filt_softmax)

            dice_ent[center].append((dice_hard, ent))

        dice_ent[center] = np.array(dice_ent[center])

        plt.scatter(dice_ent[center][:, 1], dice_ent[center][:, 0], label=center)

    plt.xlabel("Entropy")
    plt.ylabel("Dice score")

    # plt.xlim([0, 0.2 + np.max(dice_ent[center][:, 1])])
    plt.ylim([0, 1])

    plt.legend()
    if alt_title:
        plt.title(alt_title)

    plt.show()


def uncertainty_by_condition(
    path_csvs: str | None = None,
    alt_title: str = "",
    n_samples: int | None = None,
    centers: list[str] | None = None,
) -> None:
    """Uncertainty by condition

    Plots inspired in: uncertainty_by_condition from
    https://github.com/SteffenCzolbe/probabilistic_segmentation
    This will plot the TP, FP, TN, FN uncertainty for each center

    :return:
    """
    centers = centers or DEFAULT_CENTERS

    unc_cond = {}
    for center in centers:
        df = _load_center_csv(path_csvs, center)

        unc_cond[center] = {
            "TP": np.array([]),
            "FP": np.array([]),
            "TN": np.array([]),
            "FN": np.array([]),
        }

        for i, row in df.iterrows():
            pred_hard_path, pred_softmax_path, _, gt_path, _, _, _, _ = row

            subj_path = os.path.dirname(pred_softmax_path)

            pred_softmax = nib.load(pred_softmax_path).get_fdata()
            gt = nib.load(gt_path).get_fdata()
            b_mask = nib.load(get_b_mask_path(subj_path)).get_fdata()

            gt_one_hot = np.eye(2, dtype=np.uint8)[gt.astype(int)]

            neg_sftmx = pred_softmax[:, :, :, 0].flatten()
            pos_sftmx = pred_softmax[:, :, :, 1].flatten()
            neg_gt = gt_one_hot[:, :, :, 0].flatten()
            pos_gt = gt_one_hot[:, :, :, 1].flatten()

            b_mask = b_mask.flatten()

            # Brain mask
            thres_brain = np.where(b_mask == 1)[0]
            pos_brain = pos_sftmx[thres_brain]
            neg_brain = neg_sftmx[thres_brain]
            pos_gt_brain = pos_gt[thres_brain]
            neg_gt_brain = neg_gt[thres_brain]

            # TP
            tp = np.where((pos_brain >= 0.5) & (pos_gt_brain == 1))[0]
            tp_unc = entropy(pos_brain[tp], apply_mean=False)
            unc_cond[center]["TP"] = np.append(unc_cond[center]["TP"], tp_unc)

            # FP
            fp = np.where((pos_brain >= 0.5) & (pos_gt_brain == 0))[0]
            fp_unc = entropy(pos_brain[fp], apply_mean=False)
            unc_cond[center]["FP"] = np.append(unc_cond[center]["FP"], fp_unc)

            # TN
            tn = np.where((neg_brain >= 0.5) & (neg_gt_brain == 1))[0]
            tn_unc = entropy(neg_brain[tn], apply_mean=False)
            unc_cond[center]["TN"] = np.append(unc_cond[center]["TN"], tn_unc)

            # FN
            fn = np.where((neg_brain >= 0.5) & (neg_gt_brain == 0))[0]
            fn_unc = entropy(neg_brain[fn], apply_mean=False)
            unc_cond[center]["FN"] = np.append(unc_cond[center]["FN"], fn_unc)

        # Since there are too many points on TN, we will sample all categories
        # for having just 100 samples
        if n_samples:
            for category in unc_cond[center].keys():
                if len(unc_cond[center][category]) > n_samples:
                    unc_cond[center][category] = np.random.choice(
                        unc_cond[center][category], n_samples, replace=False
                    )

    # Plot Stripplot
    fig, ax = plt.subplots(len(centers), 1, figsize=(4, 10))
    categories = ["TP", "FP", "TN", "FN"]
    for i, center in enumerate(centers):
        sns.stripplot(data=unc_cond[center], ax=ax[i], jitter=True, alpha=0.05)
        ax[i].set_title(center)
        ax[i].set_ylabel("Entropy")

        ax[i].set_xticks(range(len(categories)))
        ax[i].set_xticklabels(categories)

        # Calculate mean and median for each category
        mean_values = [np.mean(unc_cond[center][category]) for category in categories]
        median_values = [np.median(unc_cond[center][category]) for category in categories]

        # Add mean and median markers for each category
        for j, category in enumerate(categories):
            ax[i].plot(
                j,
                mean_values[j],
                marker="o",
                color="red",
                label="Mean" if j == 0 else "",
                zorder=10,
            )
            ax[i].plot(
                j,
                median_values[j],
                marker="o",
                color="blue",
                label="Median" if j == 0 else "",
                zorder=10,
            )

        # Add legend
        ax[i].legend()

    if alt_title:
        fig.suptitle(alt_title)

    plt.show()
