import os.path

import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from calibration import get_ece
from medpy.metric.binary import dc as dice_score
# from scipy.stats import entropy
from sklearn.calibration import calibration_curve

sns.set_style('whitegrid')

colors = ['#3498db', '#2ecc71', '#e74c3c']  # colors for each center
centers = ['utrecht', 'singapore', 'amsterdam']
csv_cols = ['pred_wmh_hard', 'pred_wmh_softmax', 'pred_logits', 'gt_wmh']


def get_array_from_nifti(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice_scores(base_center):
    dc = {}
    for center in centers:
        df = pd.read_csv(f"{base_center}-{center}.csv", header=None)
        df.columns = csv_cols
        dc[center] = {'hard': []}

        for _, row in df.iterrows():
            pred_hard_path, pred_softmax_path, logits_path, gt_path = row

            pred_hard = nib.load(pred_hard_path).get_fdata()
            gt = nib.load(gt_path).get_fdata()

            # Compute dice scores
            dice_hard = dice_score(pred_hard, gt)

            # Save results in dictionary
            dc[center]['hard'].append(dice_hard)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), sharey=True)
    for i, center in enumerate(centers):
        sns.boxplot(y=dc[center]['hard'], color=colors[i], ax=axs[i],
                    showmeans=True)
        axs[i].set_title(f"{center} - Hard")

    fig.suptitle('Dice Scores for WMH Segmentation')
    fig.text(0.5, 0.04, 'Center', ha='center')
    fig.text(0.04, 0.5, 'Dice Score', va='center', rotation='vertical')

    fig.subplots_adjust(top=0.85, wspace=0.05)


def entropy(probs):
    """ Compute entropy for a flattened array of probabilities

    :param probs: array of probabilities
    :return:
    """

    return -np.mean(probs * np.log(probs))


def entropy_segment(base_center):
    ent = {}

    for center in centers:
        df = pd.read_csv(f"{base_center}-{center}.csv", header=None)
        df.columns = csv_cols
        ent[center] = {}

        for _, row in df.iterrows():
            pred_softmax_path, gt_path = row[1], row[3]
            subj = os.path.basename(os.path.dirname(row[1]))
            ent[center][subj] = {'pred': [], 'gt': []}

            pred_softmax = nib.load(pred_softmax_path).get_fdata()
            pos_class = pred_softmax[:, :, :, 1].flatten()

            thres_pos = np.where(pos_class >= 0.5)[0]
            filt_softmax = pos_class[thres_pos]

            gt = nib.load(gt_path).get_fdata().flatten()
            gt_1 = np.where(gt == 1)[0]
            thres_gt = pos_class[gt_1]

            entropy_pred = entropy(filt_softmax)
            entropy_gt = entropy(thres_gt)

            ent[center][subj]['pred'].append(entropy_pred)
            ent[center][subj]['gt'].append(entropy_gt)

    # Print the entropy for each center and individual
    for center in centers:
        print(f"Center: {center}")
        for key, value in ent[center].items():
            print(f"{key}: {value}")

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), sharey=True)
    for i, center in enumerate(centers):
        values = list(ent[center].values())
        values_pred = [x['pred'][0] for x in values]

        # Plot the boxplots for softmax predicted entropy
        axs[i].boxplot(values_pred, positions=[0], widths=0.6)

        axs[i].set_xticklabels(['Pred'])
        axs[i].set_title(f"{center}")


def append_round(imgs, img, decimals=2):
    return np.concatenate((imgs, np.around(np.array(img), decimals)))


def probs_hist(base_center):
    for center in centers:
        imgs_smx_0_0, imgs_smx_0_1, imgs_smx_1_0, imgs_smx_1_1 = \
            np.array([]), np.array([]), np.array([]), np.array([])

        df = pd.read_csv(f"{base_center}-{center}.csv", header=None)
        df.columns = csv_cols

        for _, row in df.iterrows():
            _, pred_wmh_softmax, _, gt_wmh = row
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
        axs[0, 0].set_yscale('log')
        axs[0, 0].set_title(f"{center} - Class 0 - GT 0")
        axs[0, 1].hist(imgs_smx_1_0, bins=20)
        axs[0, 1].set_yscale('log')
        axs[0, 1].set_title(f"{center} - Class 1 - GT 0")
        axs[1, 0].hist(imgs_smx_0_1, bins=20)
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_title(f"{center} - Class 0 - GT 1")
        axs[1, 1].hist(imgs_smx_1_1, bins=20)
        axs[1, 1].set_yscale('log')
        axs[1, 1].set_title(f"{center} - Class 1 - GT 1")


def logits_hist(base_center):
    for center in centers:
        imgs_logits_0_0, imgs_logits_0_1, imgs_logits_1_0, imgs_logits_1_1 = \
            np.array([]), np.array([]), np.array([]), np.array([])

        df = pd.read_csv(f"{base_center}-{center}.csv", header=None)
        df.columns = csv_cols

        for _, row in df.iterrows():
            _, _, pred_logits, gt_wmh = row
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

        max_val = 100

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


def ece_reliability(base_center):
    for center in centers:
        df = pd.read_csv(f"{base_center}-{center}.csv", header=None)
        df.columns = csv_cols

        preds_0 = np.array([])
        preds_1 = np.array([])
        gt_0 = np.array([], dtype=np.uint8)
        gt_1 = np.array([], dtype=np.uint8)

        for i, row in df.iterrows():
            _, pred_wmh_softmax, _, gt_wmh = row

            pred_softmax = get_array_from_nifti(pred_wmh_softmax)
            gt_wmh = get_array_from_nifti(gt_wmh)

            # One-hot encode the ground truth labels
            gt_one_hot = np.eye(2, dtype=np.uint8)[gt_wmh.astype(int)]

            # Flatten the arrays and append to numpy array
            pos_indices_0 = np.where(gt_one_hot[:, :, :, 0].flatten() == 1)[0]
            pos_indices_1 = np.where(gt_one_hot[:, :, :, 1].flatten() == 1)[0]
            preds_0 = np.concatenate((preds_0, pred_softmax[0].flatten()[pos_indices_0]))
            preds_1 = np.concatenate((preds_1, pred_softmax[1].flatten()[pos_indices_1]))
            gt_0 = np.concatenate((gt_0, gt_one_hot[:, :, :, 0].flatten()[pos_indices_0]))
            gt_1 = np.concatenate((gt_1, gt_one_hot[:, :, :, 1].flatten()[pos_indices_1]))

        # Define the number of bins for the reliability diagram
        num_bins = 10

        # Calculate the bin edges
        bin_edges = np.linspace(0, 1, num_bins + 1)

        emp_probs_0, pred_probs_0 = calibration_curve(gt_0, preds_0,
                                                      n_bins=num_bins)
        emp_probs_1, pred_probs_1 = calibration_curve(gt_1, preds_1,
                                                      n_bins=num_bins)

        ece_0 = get_ece(preds_0, gt_0)
        ece_1 = get_ece(preds_1, gt_1)

        # Plot the reliability diagram
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        ax.plot(pred_probs_0, emp_probs_0, 's-', label='Background')
        ax.plot(pred_probs_1, emp_probs_1, 'o-', label='Foreground')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Empirical probability')
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        ax.set_title(f"{center} - Reliability diagram\nECE: {ece_0:.4f} (BG), {ece_1:.4f} (FG)")

    plt.show()
