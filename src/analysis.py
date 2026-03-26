from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from calibration import get_ece
from medpy.metric.binary import dc as dice_score
from scipy.spatial.distance import directed_hausdorff
from sklearn.calibration import calibration_curve

from src.plot import entropy, get_b_mask_path

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_gt_paths(test_splits: list) -> list[str]:
    """Return subject directory paths from a test-split list.

    Each entry in test_splits is expected to be [t1_path, flair_path, gt_path]
    (or similar); the subject dir is the parent of the third element.
    """
    return [os.path.dirname(test_splits[i][2]) for i in range(len(test_splits))]


def _cached_compute(
    compute_fn,
    cache_path: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load from CSV cache if it exists, otherwise compute and optionally save."""
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    df = compute_fn(**kwargs)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def _calculate_hausdorff(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    voxel_spacing: tuple | list | None = None,
) -> float:
    """Bidirectional Hausdorff distance between two binary masks."""
    coords_gt = np.argwhere(gt_mask > 0)
    coords_pred = np.argwhere(pred_mask > 0)

    if coords_gt.size == 0 and coords_pred.size == 0:
        return 0.0

    if coords_gt.size == 0 or coords_pred.size == 0:
        dims = np.array(gt_mask.shape, dtype=float)
        if voxel_spacing is not None and len(voxel_spacing) == gt_mask.ndim:
            dims = dims * np.array(voxel_spacing)
        return float(np.linalg.norm(dims))

    if voxel_spacing is not None and len(voxel_spacing) == gt_mask.ndim:
        coords_gt = coords_gt * np.array(voxel_spacing)
        coords_pred = coords_pred * np.array(voxel_spacing)

    hd1 = directed_hausdorff(coords_gt, coords_pred)[0]
    hd2 = directed_hausdorff(coords_pred, coords_gt)[0]
    return float(max(hd1, hd2))


def _dice_score_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dice coefficient between two binary arrays."""
    inter = float(np.sum(y_true * y_pred))
    s = float(np.sum(y_true) + np.sum(y_pred))
    return 1.0 if s == 0 else 2.0 * inter / s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_plot_data(
    data_root: str,
    centers: str = "UtAmSi",
    split_ratios: list[float] | None = None,
    seed: int = 42,
) -> dict:
    """Build the common plot-data dict used by all compute functions.

    Creates WMHDataModule instances for Utrecht, Amsterdam, Singapore (in-dist)
    and UMCL (out-of-dist), generates test splits, and groups in-dist centers
    under a single key.

    Args:
        data_root: Path to the WMH dataset root directory.
        centers: Training-center group name (default ``"UtAmSi"``).
        split_ratios: Train/val/test ratios (default ``[0.7, 0.1, 0.2]``).
        seed: Random seed for split generation.

    Returns:
        dict with keys ``centers_train``, ``runs_to_compare``, ``centers_test``,
        ``test_splits``, ``losses``.  ``runs_to_compare`` is an empty dict —
        callers must populate it with ``{"{loss} {tr_center}": "run_name"}``
        entries.

    Raises:
        FileNotFoundError: If *data_root* does not exist.
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    if split_ratios is None:
        split_ratios = [0.7, 0.1, 0.2]

    from src.datamodules import WMHDataModule  # heavy Lightning dep — lazy import

    in_dist_centers = ["Utrecht", "Amsterdam", "Singapore"]
    losses = ["CE", "CE_MEEP", "CE_KL", "CE_MEALL"]

    test_splits: dict[str, list] = {}
    centers_test = list(in_dist_centers) + ["UMCL"]

    for center_name in in_dist_centers:
        dm = WMHDataModule(data_root, 1, f"training:{center_name}", split_ratios, seed=seed)
        _, _, ts_spl = dm.generate_splits()
        test_splits[center_name] = ts_spl

    test_splits["UMCL"] = WMHDataModule(data_root, 1, "training:UMCL", [0, 0, 1]).generate_splits()[
        2
    ]

    # Group in-dist centers under the combined key (e.g. "UtAmSi")
    test_splits[centers] = []
    for c in in_dist_centers:
        test_splits[centers] += test_splits.pop(c)
        centers_test.remove(c)
    centers_test.append(centers)

    return {
        "centers_train": [centers],
        "runs_to_compare": {},
        "centers_test": centers_test,
        "test_splits": test_splits,
        "losses": losses,
    }


def dice_vs_entropy_data(
    plot_data: dict,
    runs_to_compare: dict[str, str],
    entropy_mask: str = "softmax_pos_class",
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Compute per-subject Dice and Entropy for fig3.

    Args:
        plot_data: Dict from :func:`build_plot_data`.
        runs_to_compare: Mapping ``"{loss} {tr_center}" -> run_name``.
        entropy_mask: Which voxels to use for entropy.  Options:
            ``"softmax_pos_class"`` (predicted positive, >= 0.5),
            ``"gt"`` (ground-truth positive),
            ``"brain_mask"``.
        cache_path: Optional CSV path for caching.

    Returns:
        DataFrame with columns ``[Loss, Test_Center, Dice, Entropy]``.
    """

    def _compute(
        plot_data: dict,
        runs_to_compare: dict[str, str],
        entropy_mask: str,
    ) -> pd.DataFrame:
        centers_train = plot_data["centers_train"]
        centers_test = plot_data["centers_test"]
        test_splits = plot_data["test_splits"]
        losses = plot_data["losses"]

        results = []
        for tr_center in centers_train:
            for loss in losses:
                run_name = runs_to_compare.get(f"{loss} {tr_center}")
                if run_name is None:
                    continue
                for ts_center in centers_test:
                    gt_paths = _get_gt_paths(test_splits[ts_center])
                    for subj_path in gt_paths:
                        gt_p = os.path.join(subj_path, f"gt_wmh_{run_name}.nii.gz")
                        softmax_p = os.path.join(subj_path, f"pred_wmh_softmax_{run_name}.nii.gz")
                        hard_p = os.path.join(subj_path, f"pred_wmh_hard_{run_name}.nii.gz")

                        if not (
                            os.path.exists(gt_p)
                            and os.path.exists(softmax_p)
                            and os.path.exists(hard_p)
                        ):
                            continue

                        gt = nib.load(gt_p).get_fdata()
                        pred_softmax = nib.load(softmax_p).get_fdata()
                        pred_hard = nib.load(hard_p).get_fdata()

                        pos_class = pred_softmax[:, :, :, 1].flatten()

                        if entropy_mask == "softmax_pos_class":
                            filt = pos_class >= 0.5
                        elif entropy_mask == "gt":
                            filt = gt.flatten() == 1
                        else:  # brain_mask
                            b_mask = nib.load(get_b_mask_path(subj_path)).get_fdata()
                            filt = b_mask.flatten() == 1

                        sel = pos_class[filt]
                        ent = float(entropy(sel)) if sel.size > 0 else 0.0
                        dc = dice_score(pred_hard, gt)

                        results.append(
                            {
                                "Loss": loss,
                                "Test_Center": ts_center,
                                "Dice": dc,
                                "Entropy": ent,
                            }
                        )

        return pd.DataFrame(results)

    return _cached_compute(
        _compute,
        cache_path=cache_path,
        plot_data=plot_data,
        runs_to_compare=runs_to_compare,
        entropy_mask=entropy_mask,
    )


def confusion_entropy_data(
    plot_data: dict,
    runs_to_compare: dict[str, str],
    n_samples: int | None = 1000,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Compute voxel-level entropy per confusion category (TP/FP/TN/FN) for fig4.

    Args:
        plot_data: Dict from :func:`build_plot_data`.
        runs_to_compare: Mapping ``"{loss} {tr_center}" -> run_name``.
        n_samples: Max voxels to keep per (loss, center, category).  ``None`` keeps all.
        cache_path: Optional CSV path for caching.

    Returns:
        DataFrame with columns ``[Loss, Center, Distribution, Category, Entropy]``.
    """
    rename_centers = {"UtAmSi": "In-distribution", "UMCL": "Out-of-distribution"}

    def _compute(
        plot_data: dict,
        runs_to_compare: dict[str, str],
        n_samples: int | None,
    ) -> pd.DataFrame:
        centers_train = plot_data["centers_train"]
        centers_test = plot_data["centers_test"]
        test_splits = plot_data["test_splits"]
        losses = plot_data["losses"]

        rows = []
        for tr_center in centers_train:
            for loss in losses:
                run_name = runs_to_compare.get(f"{loss} {tr_center}")
                if run_name is None:
                    continue
                for ts_center in centers_test:
                    dist = rename_centers.get(ts_center, ts_center)
                    cat_arrays: dict[str, np.ndarray] = {
                        "TP": np.array([]),
                        "FP": np.array([]),
                        "TN": np.array([]),
                        "FN": np.array([]),
                    }

                    for subj_path in _get_gt_paths(test_splits[ts_center]):
                        softmax_p = os.path.join(subj_path, f"pred_wmh_softmax_{run_name}.nii.gz")
                        gt_p = os.path.join(subj_path, f"gt_wmh_{run_name}.nii.gz")
                        b_mask_p = get_b_mask_path(subj_path)

                        if not (
                            os.path.exists(softmax_p)
                            and os.path.exists(gt_p)
                            and os.path.exists(b_mask_p)
                        ):
                            continue

                        pred_softmax = nib.load(softmax_p).get_fdata()
                        gt = nib.load(gt_p).get_fdata()
                        b_mask = nib.load(b_mask_p).get_fdata()

                        gt_one_hot = np.eye(2, dtype=np.uint8)[gt.astype(int)]
                        pos_sftmx = pred_softmax[:, :, :, 1].flatten()
                        neg_sftmx = pred_softmax[:, :, :, 0].flatten()
                        pos_gt = gt_one_hot[:, :, :, 1].flatten()
                        neg_gt = gt_one_hot[:, :, :, 0].flatten()
                        b_flat = b_mask.flatten()

                        brain_idx = np.where(b_flat == 1)[0]
                        pos_brain = pos_sftmx[brain_idx]
                        neg_brain = neg_sftmx[brain_idx]
                        pos_gt_brain = pos_gt[brain_idx]
                        neg_gt_brain = neg_gt[brain_idx]

                        tp_idx = np.where((pos_brain >= 0.5) & (pos_gt_brain == 1))[0]
                        fp_idx = np.where((pos_brain >= 0.5) & (pos_gt_brain == 0))[0]
                        tn_idx = np.where((neg_brain >= 0.5) & (neg_gt_brain == 1))[0]
                        fn_idx = np.where((neg_brain >= 0.5) & (neg_gt_brain == 0))[0]

                        cat_arrays["TP"] = np.append(
                            cat_arrays["TP"], entropy(pos_brain[tp_idx], apply_mean=False)
                        )
                        cat_arrays["FP"] = np.append(
                            cat_arrays["FP"], entropy(pos_brain[fp_idx], apply_mean=False)
                        )
                        cat_arrays["TN"] = np.append(
                            cat_arrays["TN"], entropy(neg_brain[tn_idx], apply_mean=False)
                        )
                        cat_arrays["FN"] = np.append(
                            cat_arrays["FN"], entropy(neg_brain[fn_idx], apply_mean=False)
                        )

                    if n_samples:
                        for cat in cat_arrays:
                            arr = cat_arrays[cat]
                            if len(arr) > n_samples:
                                cat_arrays[cat] = np.random.choice(arr, n_samples, replace=False)

                    for cat, values in cat_arrays.items():
                        for v in values:
                            rows.append(
                                {
                                    "Loss": loss,
                                    "Center": ts_center,
                                    "Distribution": dist,
                                    "Category": cat,
                                    "Entropy": float(v),
                                }
                            )

        return pd.DataFrame(rows)

    return _cached_compute(
        _compute,
        cache_path=cache_path,
        plot_data=plot_data,
        runs_to_compare=runs_to_compare,
        n_samples=n_samples,
    )


def compute_all_metrics(
    plot_data: dict,
    runs_to_compare: dict[str, str],
    entropy_mask: str = "softmax_pos_class",
    use_header_spacing: bool = True,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Compute per-subject Entropy, Dice, and Hausdorff distance for fig5.

    Args:
        plot_data: Dict from :func:`build_plot_data`.
        runs_to_compare: Mapping ``"{loss} {tr_center}" -> run_name``.
        entropy_mask: Which voxels to use for entropy (same options as
            :func:`dice_vs_entropy_data`).
        use_header_spacing: If ``True``, use voxel spacing from the NIfTI header
            when computing Hausdorff distance.
        cache_path: Optional CSV path for caching.

    Returns:
        DataFrame with columns
        ``[Center, Loss, Distribution, Entropy, Dice Score, Hausdorff Distance, Case]``.
    """
    rename_centers = {"UtAmSi": "In-distribution", "UMCL": "Out-of-distribution"}

    def _compute(
        plot_data: dict,
        runs_to_compare: dict[str, str],
        entropy_mask: str,
        use_header_spacing: bool,
    ) -> pd.DataFrame:
        centers_train = plot_data["centers_train"]
        centers_test = plot_data["centers_test"]
        test_splits = plot_data["test_splits"]
        losses = plot_data["losses"]

        records = []
        for tr_center in centers_train:
            for loss in losses:
                run_name = runs_to_compare.get(f"{loss} {tr_center}")
                if run_name is None:
                    continue
                for ts_center in centers_test:
                    dist = rename_centers.get(ts_center, ts_center)
                    for subj_path in _get_gt_paths(test_splits[ts_center]):
                        try:
                            gt_p = os.path.join(subj_path, f"gt_wmh_{run_name}.nii.gz")
                            pred_p = os.path.join(subj_path, f"pred_wmh_softmax_{run_name}.nii.gz")

                            if not (os.path.exists(gt_p) and os.path.exists(pred_p)):
                                continue

                            gt_img = nib.load(gt_p)
                            pred_img = nib.load(pred_p)
                            gt = gt_img.get_fdata()
                            pr = pred_img.get_fdata()

                            if pr.ndim > gt.ndim and pr.shape[-1] > 1:
                                pr = pr[..., 1]
                            elif pr.ndim > gt.ndim and pr.shape[-1] == 1:
                                pr = pr[..., 0]

                            vox = (
                                gt_img.header.get_zooms()[: gt.ndim] if use_header_spacing else None
                            )

                            # Entropy
                            flat = pr.flatten()
                            if entropy_mask == "softmax_pos_class":
                                sel = flat[flat >= 0.5]
                            elif entropy_mask == "gt":
                                sel = flat[gt.flatten() == 1]
                            else:  # brain_mask
                                b_mask_p = get_b_mask_path(subj_path)
                                if os.path.exists(b_mask_p):
                                    bm = nib.load(b_mask_p).get_fdata()
                                    sel = (
                                        flat[bm.flatten() == 1]
                                        if bm.shape == gt.shape
                                        else np.empty(0)
                                    )
                                else:
                                    sel = np.empty(0)

                            if sel.size > 0:
                                p = np.clip(sel, 1e-10, 1 - 1e-10)
                                ent = float((-p * np.log(p) - (1 - p) * np.log(1 - p)).mean())
                            else:
                                ent = 0.0

                            # Dice
                            bin_pr = (pr >= 0.5).astype(np.int8)
                            dsc = _dice_score_simple(gt.astype(np.int8), bin_pr)

                            # Hausdorff
                            coords_gt = np.argwhere(gt > 0)
                            coords_pred = np.argwhere(bin_pr > 0)

                            if coords_gt.size == 0 and coords_pred.size == 0:
                                case = "no_gt_no_pred"
                                hd = 0.0
                            elif coords_gt.size == 0 or coords_pred.size == 0:
                                case = "one_empty"
                                dims = np.array(gt.shape, dtype=float)
                                if vox is not None:
                                    dims = dims * np.array(vox)
                                hd = float(np.linalg.norm(dims))
                            else:
                                case = "normal"
                                hd = _calculate_hausdorff(gt, bin_pr, voxel_spacing=vox)

                            records.append(
                                {
                                    "Center": tr_center,
                                    "Loss": loss,
                                    "Distribution": dist,
                                    "Entropy": ent,
                                    "Dice Score": dsc,
                                    "Hausdorff Distance": hd,
                                    "Case": case,
                                }
                            )
                        except Exception:
                            continue

        return pd.DataFrame(records)

    return _cached_compute(
        _compute,
        cache_path=cache_path,
        plot_data=plot_data,
        runs_to_compare=runs_to_compare,
        entropy_mask=entropy_mask,
        use_header_spacing=use_header_spacing,
    )


def entropy_volume_ranges(
    plot_data: dict,
    runs_to_compare: dict[str, str],
    volume_ranges: tuple[tuple[float, float], ...] = ((0, 5), (5, 15), (15, float("inf"))),
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Compute per-lesion entropy grouped by predicted lesion volume for fig6.

    Connected components of the predicted hard mask define individual lesions.
    Volume is computed in physical units using the image spacing.

    Args:
        plot_data: Dict from :func:`build_plot_data`.
        runs_to_compare: Mapping ``"{loss} {tr_center}" -> run_name``.
        volume_ranges: Sequence of ``(low, high)`` volume bins (mm³).  A lesion
            falls in the first range where ``low < volume <= high``.
        cache_path: Optional CSV path for caching.

    Returns:
        DataFrame with columns ``[Loss, Center, Volume Range, Entropy]``.
    """
    rename_centers = {"UtAmSi": "In-distribution", "UMCL": "Out-of-distribution"}

    def _compute(
        plot_data: dict,
        runs_to_compare: dict[str, str],
        volume_ranges: tuple,
    ) -> pd.DataFrame:
        centers_train = plot_data["centers_train"]
        centers_test = plot_data["centers_test"]
        test_splits = plot_data["test_splits"]
        losses = plot_data["losses"]

        rows = []
        for tr_center in centers_train:
            for loss in losses:
                run_name = runs_to_compare.get(f"{loss} {tr_center}")
                if run_name is None:
                    continue
                for ts_center in centers_test:
                    for subj_path in _get_gt_paths(test_splits[ts_center]):
                        pred_file = os.path.join(subj_path, f"pred_wmh_softmax_{run_name}.nii.gz")
                        if not os.path.exists(pred_file):
                            continue
                        try:
                            pred = sitk.ReadImage(pred_file)
                            arr = sitk.GetArrayFromImage(pred)

                            # Ensure 2-channel (softmax) layout
                            if arr.ndim != 4 or arr.shape[0] != 2:
                                continue

                            hard = arr.argmax(axis=0)
                            img_for_cc = sitk.GetImageFromArray((hard > 0).astype(np.uint8))
                            img_for_cc.SetSpacing(pred.GetSpacing()[: img_for_cc.GetDimension()])

                            cc = sitk.RelabelComponent(sitk.ConnectedComponent(img_for_cc))
                            max_label = int(sitk.GetArrayFromImage(cc).max())

                            spacing = img_for_cc.GetSpacing()
                            voxel_vol = float(np.prod(spacing))

                            for lbl in range(1, max_label + 1):
                                mask_arr = sitk.GetArrayFromImage(cc == lbl).flatten().astype(bool)
                                if not mask_arr.any():
                                    continue
                                vol = float(mask_arr.sum()) * voxel_vol
                                probs = arr[1].flatten()[mask_arr]
                                if probs.size == 0:
                                    continue
                                ent = float(entropy(probs, apply_mean=True))
                                if np.isnan(ent):
                                    continue

                                for low, high in volume_ranges:
                                    if low < vol <= high:
                                        vr_label = f"> {low}, <= {high}"
                                        rows.append(
                                            {
                                                "Loss": loss,
                                                "Center": rename_centers.get(ts_center, ts_center),
                                                "Volume Range": vr_label,
                                                "Entropy": ent,
                                            }
                                        )
                                        break
                        except Exception:
                            continue

        return pd.DataFrame(rows)

    return _cached_compute(
        _compute,
        cache_path=cache_path,
        plot_data=plot_data,
        runs_to_compare=runs_to_compare,
        volume_ranges=volume_ranges,
    )


def reliability_data(
    plot_data: dict,
    runs_to_compare: dict[str, str],
    num_bins: int = 10,
    max_voxels: int = 30_000_000,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Compute calibration curve data and ECE per (loss, center) for fig7.

    Args:
        plot_data: Dict from :func:`build_plot_data`.
        runs_to_compare: Mapping ``"{loss} {tr_center}" -> run_name``.
        num_bins: Number of bins for the calibration curve.
        max_voxels: Maximum voxels to use per (loss, center) pair (sampled
            uniformly if exceeded).
        cache_path: Optional CSV path for caching.

    Returns:
        DataFrame with columns ``[Loss, Center, Distribution, PredProb, EmpProb, ECE]``.
        Each row is one calibration-curve bin point.
    """
    rename_centers = {"UtAmSi": "In-distribution", "UMCL": "Out-of-distribution"}

    def _compute(
        plot_data: dict,
        runs_to_compare: dict[str, str],
        num_bins: int,
        max_voxels: int,
    ) -> pd.DataFrame:
        centers_train = plot_data["centers_train"]
        centers_test = plot_data["centers_test"]
        test_splits = plot_data["test_splits"]
        losses = plot_data["losses"]

        rows = []
        for tr_center in centers_train:
            for loss in losses:
                run_name = runs_to_compare.get(f"{loss} {tr_center}")
                if run_name is None:
                    continue
                for ts_center in centers_test:
                    dist = rename_centers.get(ts_center, ts_center)
                    preds_list = []
                    gts_list = []

                    for subj_path in _get_gt_paths(test_splits[ts_center]):
                        gt_p = os.path.join(subj_path, f"gt_wmh_{run_name}.nii.gz")
                        pred_p = os.path.join(subj_path, f"pred_wmh_softmax_{run_name}.nii.gz")

                        if not (os.path.exists(gt_p) and os.path.exists(pred_p)):
                            continue

                        try:
                            pred_data = nib.load(pred_p).get_fdata()
                            gt_data = nib.load(gt_p).get_fdata().astype(bool)

                            if pred_data.shape[: gt_data.ndim] != gt_data.shape:
                                continue

                            if pred_data.ndim == gt_data.ndim + 1 and pred_data.shape[-1] >= 2:
                                subj_probs = pred_data[..., 1]
                            elif pred_data.ndim == gt_data.ndim:
                                subj_probs = pred_data
                            else:
                                continue

                            preds_list.append(subj_probs.flatten())
                            gts_list.append(gt_data.flatten())
                        except Exception:
                            continue

                    if not preds_list:
                        continue

                    try:
                        preds_arr = np.concatenate(preds_list)
                        gts_arr = np.concatenate(gts_list)
                    except (ValueError, MemoryError):
                        continue

                    if len(preds_arr) > max_voxels:
                        idx = np.random.choice(len(preds_arr), max_voxels, replace=False)
                        preds_arr = preds_arr[idx]
                        gts_arr = gts_arr[idx]

                    if len(gts_arr) == 0:
                        continue

                    try:
                        emp_probs, pred_probs_bins = calibration_curve(
                            gts_arr.astype(bool), preds_arr, n_bins=num_bins, strategy="uniform"
                        )
                        ece_val = float(get_ece(preds_arr, gts_arr.astype(int)))
                    except Exception:
                        continue

                    for ep, pp in zip(emp_probs, pred_probs_bins):
                        rows.append(
                            {
                                "Loss": loss,
                                "Center": ts_center,
                                "Distribution": dist,
                                "PredProb": float(pp),
                                "EmpProb": float(ep),
                                "ECE": ece_val,
                            }
                        )

        return pd.DataFrame(rows)

    return _cached_compute(
        _compute,
        cache_path=cache_path,
        plot_data=plot_data,
        runs_to_compare=runs_to_compare,
        num_bins=num_bins,
        max_voxels=max_voxels,
    )
