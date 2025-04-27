import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from calibration import get_ece
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D  # For custom legend elements
from medpy.metric.binary import dc as dice_score
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import linregress, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc
from tabulate import tabulate


def get_gt_paths(test_splits):
    """ Get the ground truth paths for a given test split """
    return [os.path.split(test_splits[i][2])[0] for i in
            range(len(test_splits))]


def entropy(probs, eps=1e-3, apply_mean=True):
    probs = np.clip(probs, eps, 1 - eps)
    if apply_mean:
        return np.mean(
            -probs * np.log(probs) - (1 - probs) * np.log(1 - probs))
    else:
        return -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)


def get_b_mask_path(subj_path):
    """ Load brain mask

    Given a subject path, check if the brain mask exists. If it does, return its
    path. If it doesn't, create it (using FSLs BET) and return its path.

    :param subj_path: Path to subject folder
    :return: Brain mask path
    """
    b_mask_path = os.path.join(subj_path, 'pre', 'T1_brain_mask.nii.gz')
    if os.path.exists(b_mask_path):
        return b_mask_path
    else:
        t1_path = os.path.join(subj_path, 'pre', 'T1.nii.gz')
        b_t1b_path = os.path.join(subj_path, 'pre', 'T1_brain.nii.gz')
        cmd = f"bet {t1_path} {b_t1b_path} -m"
        os.system(cmd)
        print(f"Created brain mask for {subj_path}")
        return b_mask_path


def load_images(subj_path, run_name, b_mask_path):
    """Load necessary images for processing.
    
    Args:
        subj_path: Path to the subject folder
        run_name: Name of the run
        b_mask_path: Path to the brain mask
    
    Returns:
        Tuple containing (gt, pred_softmax, pred_hard, b_mask)
    """
    gt_path = os.path.join(subj_path, f'gt_wmh_{run_name}.nii.gz')
    pred_softmax_path = os.path.join(subj_path, f'pred_wmh_softmax_{run_name}.nii.gz')
    pred_hard_path = os.path.join(subj_path, f'pred_wmh_hard_{run_name}.nii.gz')
    
    gt = nib.load(gt_path).get_fdata()
    pred_softmax = nib.load(pred_softmax_path).get_fdata()
    pred_hard = nib.load(pred_hard_path).get_fdata()
    b_mask = nib.load(b_mask_path).get_fdata()
    
    return gt, pred_softmax, pred_hard, b_mask


def flatten_and_filter(pred_softmax, pred_hard, gt, b_mask, entropy_mask='softmax_pos_class', dice_mask=None):
    """Flatten and filter the images based on the entropy mask and dice mask.
    
    Args:
        pred_softmax: Predicted softmax probabilities
        pred_hard: Hard predictions
        gt: Ground truth
        b_mask: Brain mask
        entropy_mask: Type of mask to use for entropy calculation
        dice_mask: Type of mask to use for dice calculation
    
    Returns:
        Tuple containing (pos_class, hard_class, gt_class, b_mask)
    """
    pos_class = pred_softmax[:, :, :, 1].flatten()
    hard_class = pred_hard.flatten()
    gt_class = gt.flatten()
    b_mask = b_mask.flatten()
    
    if entropy_mask == 'softmax_pos_class':
        filt = np.where(pos_class >= 0.5)[0]
        pos_class = pos_class[filt]
    elif entropy_mask == 'gt':
        filt = np.where(gt_class == 1)[0]
        pos_class = pos_class[filt]
    else:  # brain_mask
        filt = np.where(b_mask == 1)[0]
        pos_class = pos_class[filt]
    
    if dice_mask == 'brain_mask':
        filt_brain = np.where(b_mask == 1)[0]
        hard_class = hard_class[filt_brain]
        gt_class = gt_class[filt_brain]
    
    return pos_class, hard_class, gt_class, b_mask


def dice_vs_entropy(plot_data, entropy_mask='softmax_pos_class',
                    loss_centers_tendencies=False,
                    loss_tendencies=True, print_ideal=False, dice_mask=None,
                    logit_dice=False):
    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()
    results = []

    test_center_rename = {
        "UtAmSi": "In-distribution",
        "UMCL": "Out-of-distribution"
    }

    for tr_center in centers_train:
        for loss in losses:
            run_name = runs_to_compare.get(f'{loss} {tr_center}')
            if run_name is None:
                print(f'No run for {loss} {tr_center}.')
                continue

            for ts_center in centers_test:
                gt_paths = get_gt_paths(test_splits[ts_center])
                for subj_path in gt_paths:
                    gt, pred_softmax, pred_hard, b_mask = load_images(
                        subj_path, run_name, get_b_mask_path(subj_path)
                    )

                    pos_class, hard_class, gt_class, b_mask = flatten_and_filter(
                        pred_softmax, pred_hard, gt, b_mask, entropy_mask,
                        dice_mask
                    )

                    dice_hard = dice_score(hard_class, gt_class)
                    entrop = entropy(pos_class)
                    results.append({
                        'Loss': loss, 'Test_Center': ts_center,
                        'Dice': dice_hard, 'Entropy': entrop
                    })
    df = pd.DataFrame(results)

    unique_centers = df['Test_Center'].unique()
    unique_losses = df['Loss'].unique()

    fig, ax = plt.subplots(figsize=(6, 5))  # Reduced figure size

    if logit_dice:
        df['Logit_Dice'] = np.log(df['Dice'] / (1 - df['Dice']))
        scatter = sns.scatterplot(data=df, x='Logit_Dice', y='Entropy',
                                  hue='Loss', style='Test_Center', s=50)
        ax.set_xlabel('Logit(Dice Coefficient)', fontsize=12)
    else:
        scatter = sns.scatterplot(data=df, x='Dice', y='Entropy', hue='Loss',
                                  style='Test_Center', s=50)
        ax.set_xlabel('Dice Coefficient', fontsize=12)

    if print_ideal:
        ax.plot([0, 1], [1, 0], linestyle='--', color='black', linewidth=1,
                label='Ideal')

    if loss_centers_tendencies:
        for loss in losses:
            for center in centers_test:
                subset = df[
                    (df['Loss'] == loss) & (df['Test_Center'] == center)]
                if not subset.empty:
                    sns.regplot(x='Dice' if not logit_dice else 'Logit_Dice',
                                y='Entropy', data=subset, scatter=False,
                                ci=None, ax=ax, line_kws={'linewidth': 1})

    if loss_tendencies:
        for loss in losses:
            subset = df[df['Loss'] == loss]
            if not subset.empty:
                sns.regplot(x='Dice' if not logit_dice else 'Logit_Dice',
                            y='Entropy', data=subset, scatter=False, ci=None,
                            ax=ax, line_kws={'linewidth': 1})

    ax.set_title('Dice vs. Entropy', fontsize=14, pad=10)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    if not logit_dice:
        ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Remove the default legend
    ax.get_legend().remove()

    # Calculate Spearman correlations and find the maximum length of loss names for alignment
    max_len = max(len(loss.split('_')[1] if '_' in loss else loss) for loss in
                  unique_losses)

    # Create custom legend elements
    palette = sns.color_palette(n_colors=len(unique_losses))
    loss_elements = []
    for i, loss in enumerate(unique_losses):
        subset = df[df['Loss'] == loss]
        if not subset.empty:
            corr, _ = pearsonr(subset['Dice'], subset['Entropy'])
            corr, _ = spearmanr(subset['Dice'], subset['Entropy'])
            loss_name = loss.split('_')[1] if '_' in loss else loss
            padded_loss_name = f"{loss_name}{' ' * (max_len - len(loss_name))}"  # Adding this padding doesn't work because they're still disaligned
            loss_label = f"{padded_loss_name} ({corr:.3f})"
            loss_elements.append(
                plt.Line2D([0], [0], marker='s', color=palette[i], linestyle='',
                           markersize=8, label=loss_label))

    center_elements = [
        plt.Line2D([0], [0], marker='o' if center == 'UMCL' else 'x',
                   color='black', linestyle='', markersize=8, label=center)
        for center in unique_centers]

    if rename_centers:
        for i, center in enumerate(center_elements):
            center_elements[i].set_label(
                rename_centers.get(center.get_label(), center.get_label()))
    # Sort center elements alphabetically
    center_elements = sorted(center_elements, key=lambda x: x.get_label())

    # Add empty elements to ensure column break
    num_loss = len(loss_elements)
    num_center = len(center_elements)
    max_elements = max(num_loss, num_center)

    loss_elements = [plt.Line2D([0], [0], marker='', linestyle='',
                                label='Loss')] + loss_elements
    loss_elements += [plt.Line2D([0], [0], marker='', linestyle='',
                                 label='')] * (max_elements - num_loss)
    center_elements = [plt.Line2D([0], [0], marker='', linestyle='',
                                  label='Data')] + center_elements
    center_elements += [plt.Line2D([0], [0], marker='', linestyle='',
                                   label='')] * (max_elements - num_center)

    # Combine all legend elements
    all_elements = loss_elements + center_elements

    # Create a single legend with two columns
    legend = ax.legend(handles=all_elements, fontsize=8,
                       bbox_to_anchor=(1.05, 1), loc='upper left',
                       ncol=2, columnspacing=1, handletextpad=0.5)
    for text in legend.get_texts():
        text.set_fontname('DejaVu Sans Mono')
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Adjust right margin for legend

    plt.show()


def dice(plot_data, use_brain_mask=False):
    """ Plot dice scores for each run

    :param plot_data: Dictionary with the required data for plotting (same
    for all plotting functions)
    """

    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:  # Three plots: one per center used for training
        plot_data = {}  # Dice scores for each run
        for loss in losses:  # One subplot per loss
            plot_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:  # For each center used for testing
                plot_data[loss][ts_center] = []
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # Lst of tst sbj pts
                for subj_path in gt_pths:
                    gt_path = os.path.join(subj_path,
                                           f'gt_wmh_{run_name}.nii.gz')
                    pred_hard_path = os.path.join(subj_path,
                                                  f'pred_wmh_hard_{run_name}.nii.gz')

                    pred_hard = nib.load(pred_hard_path).get_fdata()
                    gt = nib.load(gt_path).get_fdata()
                    if use_brain_mask:
                        b_mask = nib.load(
                            get_b_mask_path(subj_path)).get_fdata()
                        # Mask pred_hard keeping the shape
                        pred_hard = pred_hard * b_mask

                    plot_data[loss][ts_center].append(
                        dice_score(pred_hard, gt))
        df = pd.DataFrame(plot_data)

        fig, axes = plt.subplots(1, len(losses), figsize=(16, 2), sharey=True)
        plt.subplots_adjust(wspace=0.1)

        # Create a boxplot for each subplot
        for i, loss in enumerate(losses):
            method_subset = df[loss]
            # # Add to method subset another element with the all the dice scores
            # method_subset['All'] = [dsc for subl in [lst for lst in method_subset] for dsc in subl]

            ax = sns.boxplot(data=method_subset, ax=axes[i])
            ax.set_title(loss)
            ax.set_xticklabels(method_subset.keys())  # set the xticklabels
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
            ax.set_ylim([0, 1])

        plt.suptitle(f'Dice scores using {tr_center} for training')
        plt.subplots_adjust(top=0.75)

        plt.show()


def hausdorff(plot_data, use_brain_mask=False):
    """Plot Hausdorff distance for each run

    :param plot_data: Dictionary with the required data for plotting
    :param use_brain_mask: If True, use brain mask to filter predictions
    """

    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:
        plot_data = {}  # Hausdorff distance for each run
        for loss in losses:
            plot_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:
                plot_data[loss][ts_center] = []
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # List of test subject paths
                for subj_path in gt_pths:
                    gt_path = os.path.join(subj_path,
                                           f'gt_wmh_{run_name}.nii.gz')
                    pred_hard_path = os.path.join(subj_path,
                                                  f'pred_wmh_hard_{run_name}.nii.gz')

                    pred_hard = nib.load(pred_hard_path).get_fdata()
                    gt = nib.load(gt_path).get_fdata()
                    if use_brain_mask:
                        b_mask = nib.load(
                            get_b_mask_path(subj_path)).get_fdata()
                        # Mask pred_hard keeping the shape
                        pred_hard = pred_hard * b_mask

                    # Calculate Hausdorff distance
                    hausdorff_dist = \
                        directed_hausdorff(np.argwhere(gt),
                                           np.argwhere(pred_hard))[
                            0]
                    plot_data[loss][ts_center].append(hausdorff_dist)

        df = pd.DataFrame(plot_data)

        fig, axes = plt.subplots(1, len(losses), figsize=(16, 2), sharey=True)
        plt.subplots_adjust(wspace=0.1)

        # Create a boxplot for each subplot
        for i, loss in enumerate(losses):
            method_subset = df[loss]
            ax = sns.boxplot(data=method_subset, ax=axes[i])
            ax.set_title(loss)
            ax.set_xticklabels(method_subset.keys())  # set the xticklabels
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
            ax.set_ylim(bottom=0)

        plt.suptitle(f'Hausdorff distances using {tr_center} for training')
        plt.subplots_adjust(top=0.75)

        plt.show()


def entropy_boxes(plot_data, m_type='softmax_pos_class'):
    """ Plot entropy scores for each run

    :param plot_data: Dictionary with the required data for plotting (same
    for all plotting functions)
    :param m_type: Type of mask to use for entropy calculation. Options:
    'softmax_pos_class', 'gt', 'brain_mask'.
    """

    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:  # Three plots: one per center used for training
        plot_data = {}  # Entropy scores for each run
        for loss in losses:  # One subplot per loss
            plot_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:  # For each center used for testing
                plot_data[loss][ts_center] = []
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # Lst of tst sbj pts
                for subj_path in gt_pths:
                    gt_path = os.path.join(subj_path,
                                           f'gt_wmh_{run_name}.nii.gz')
                    pred_softmax_path = os.path.join(subj_path,
                                                     f'pred_wmh_softmax_{run_name}.nii.gz')

                    gt = nib.load(gt_path).get_fdata()
                    pred_softmax = nib.load(pred_softmax_path).get_fdata()
                    b_mask = nib.load(
                        get_b_mask_path(subj_path)).get_fdata()

                    pos_class = pred_softmax[:, :, :, 1].flatten()
                    gt_class = gt.flatten()
                    b_mask = b_mask.flatten()

                    if m_type == 'softmax_pos_class':
                        filt = pos_class[np.where(pos_class >= 0.5)[0]]
                    elif m_type == 'gt':
                        filt = pos_class[np.where(gt_class == 1)[0]]
                    else:
                        filt = pos_class[np.where(b_mask == 1)[0]]

                    plot_data[loss][ts_center].append(entropy(filt))
        df = pd.DataFrame(plot_data)

        fig, axes = plt.subplots(1, len(losses), figsize=(16, 2),
                                 sharey=True)
        plt.subplots_adjust(wspace=0.1)

        # Create a boxplot for each subplot
        for i, loss in enumerate(losses):
            method_subset = df[loss]
            ax = sns.boxplot(data=method_subset, ax=axes[i])
            ax.set_title(loss)
            ax.set_xticklabels(method_subset.keys())  # set the xticklabels
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
            # ax.set_ylim([0, 1])

        plt.suptitle(f'Entropy using {tr_center} for training. '
                     f'Mask type: {m_type}')
        plt.subplots_adjust(top=0.75)

        plt.show()


def mcdropout_boxes(plot_data):
    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:  # Three plots: one per training center
        plot_data = {}  # Entropy scores for each run
        for loss in losses:  # One subplot per loss
            plot_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:  # For each center used for testing
                plot_data[loss][ts_center] = []
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # Lst of tst sbj pts
                for subj_path in gt_pths:
                    mc_var_path = os.path.join(subj_path,
                                               f'pred_mc_uncertmc_{run_name}.nii.gz')
                    mc_var_img = nib.load(mc_var_path).get_fdata()
                    mean_var = mc_var_img.flatten().mean()

                    plot_data[loss][ts_center].append(mean_var)
        df = pd.DataFrame(plot_data)

        fig, axes = plt.subplots(1, len(losses), figsize=(16, 2), sharey=True)
        plt.subplots_adjust(wspace=0.1)

        # Create a boxplot for each subplot
        for i, loss in enumerate(losses):
            method_subset = df[loss]
            ax = sns.boxplot(data=method_subset, ax=axes[i])
            ax.set_title(loss)
            ax.set_xticklabels(method_subset.keys())  # set the xticklabels
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
            # ax.set_ylim([0, 1])

        plt.suptitle(
            f'MC dropout estimated uncertainty using {tr_center} for training')
        plt.subplots_adjust(top=0.75)

        plt.show()


def uncertainty_confusion(plot_data, n_samples=1000):
    """ Plot uncertainty confusion matrix for each run

    :param plot_data: Dictionary with the required data for plotting (same
    for all plotting functions)
    :param n_samples: Number of samples to plot. If None, plot all samples.
    """
    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:  # Three plots: one per training center
        plot_data = {}  # Entropy scores for each run
        mean_data = {}
        median_data = {}
        for loss in losses:  # One subplot per loss
            plot_data[loss] = {}  # Three uncertainty plots per loss
            mean_data[loss] = {}
            median_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:  # For each center used for testing
                plot_data[loss][ts_center] = {
                    "TP": np.array([]),
                    "FP": np.array([]),
                    "TN": np.array([]),
                    "FN": np.array([])
                }
                mean_data[loss][ts_center] = {}
                median_data[loss][ts_center] = {}
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # List of test subject paths
                for subj_path in gt_pths:
                    pred_softmax_path = os.path.join(
                        subj_path, f'pred_wmh_softmax_{run_name}.nii.gz'
                    )
                    gt_path = os.path.join(subj_path,
                                           f'gt_wmh_{run_name}.nii.gz')

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
                    plot_data[loss][ts_center]["TP"] = np.append(
                        plot_data[loss][ts_center]["TP"], tp_unc)

                    # FP
                    fp = np.where((pos_brain >= 0.5) & (pos_gt_brain == 0))[0]
                    fp_unc = entropy(pos_brain[fp], apply_mean=False)
                    plot_data[loss][ts_center]["FP"] = np.append(
                        plot_data[loss][ts_center]["FP"], fp_unc)

                    # TN
                    tn = np.where((neg_brain >= 0.5) & (neg_gt_brain == 1))[0]
                    tn_unc = entropy(neg_brain[tn], apply_mean=False)
                    plot_data[loss][ts_center]["TN"] = np.append(
                        plot_data[loss][ts_center]["TN"], tn_unc)

                    # FN
                    fn = np.where((neg_brain >= 0.5) & (neg_gt_brain == 0))[0]
                    fn_unc = entropy(neg_brain[fn], apply_mean=False)
                    plot_data[loss][ts_center]["FN"] = np.append(
                        plot_data[loss][ts_center]["FN"], fn_unc)

                    # Since there are too many points on TN, we will sample all categories
                    # for having just 100 samples
                    if n_samples:
                        for category in plot_data[loss][ts_center].keys():
                            if len(plot_data[loss][ts_center][
                                       category]) > n_samples:
                                plot_data[loss][ts_center][
                                    category] = np.random.choice(
                                    plot_data[loss][ts_center][category],
                                    n_samples,
                                    replace=False)

                for category in ["TP", "TN", "FP", "FN"]:
                    mean_values = np.mean(plot_data[loss][ts_center][category])
                    median_values = np.median(
                        plot_data[loss][ts_center][category])
                    mean_data[loss][ts_center][category] = mean_values
                    median_data[loss][ts_center][category] = median_values

        # Continue with plotting
        fig, axes = plt.subplots(1, len(plot_data), figsize=(32, 4),
                                 sharey=True)

        # Create separate strip plots for each loss
        for i, (loss, centers_data) in enumerate(plot_data.items()):
            data = []

            for center, metrics in centers_data.items():
                for metric, values in metrics.items():
                    data.extend(
                        [(loss, center, metric, value) for value in values])

            df = pd.DataFrame(data,
                              columns=['Loss', 'Test_Center', 'Metric',
                                       'Value'])

            # Create the strip plot
            sns.stripplot(data=df, x='Metric', y='Value', hue='Test_Center',
                          palette='Dark2', jitter=True, alpha=0.2, dodge=True,
                          ax=axes[i])

            # Create the strip plot for mean_data
            mean_df = pd.DataFrame(
                [(loss, center, metric, value) for center, metrics in
                 mean_data[loss].items() for metric, value in metrics.items()],
                columns=['Loss', 'Test_Center', 'Metric', 'Value'])
            sns.stripplot(data=mean_df, x='Metric', y='Value',
                          hue='Test_Center',
                          palette='blend:#ff0000,#ff0000', marker='D', size=10,
                          dodge=True, ax=axes[i], legend=False)

            # Create the strip plot for median_data
            median_df = pd.DataFrame(
                [(loss, center, metric, value) for center, metrics in
                 median_data[loss].items() for metric, value in
                 metrics.items()],
                columns=['Loss', 'Test_Center', 'Metric', 'Value'])
            sns.stripplot(data=median_df, x='Metric', y='Value',
                          hue='Test_Center',
                          palette='blend:#0000ff,#0000ff', marker='o', size=10,
                          dodge=True, ax=axes[i], legend=False)

            axes[i].set_title(f'{loss}')
            axes[i].set_xlabel('Metric')
            axes[i].set_ylabel('Value')

            # Remove the legend from the subplot (it will be added later)
            axes[i].get_legend().remove()
            # Add legend only for the last subplot
            if i == len(plot_data) - 1:
                # Create custom legend elements for mean and median
                white_space = Line2D([0], [0], color='w', markersize=9,
                                     label=' ')
                mean_legend_element = Line2D([0], [0], marker='D', color='w',
                                             markerfacecolor='#ff0000',
                                             markersize=9, label='Mean')
                median_legend_element = Line2D([0], [0], marker='o', color='w',
                                               markerfacecolor='#0000ff',
                                               markersize=9, label='Median')

                # Add custom legend elements to the subplot
                handles, labels = axes[i].get_legend_handles_labels()
                handles.extend(
                    [white_space, mean_legend_element, median_legend_element])
                labels.extend([' ', 'Mean', ' Median'])
                axes[i].legend(handles=handles, labels=labels,
                               title='Test Center',
                               bbox_to_anchor=(1.05, 1), loc='upper left')

                # Adjust the layout
        plt.suptitle(f'Uncertainty plots using {tr_center} for training')
        plt.show()


def entropy_volume_ranges(plot_data,
                          volume_ranges=((0, 5), (5, 15), (15, np.inf))):
    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:  # Three plots: one per center used for training
        plot_data = {}  # Entropy scores for each volume range
        for loss in losses:  # One subplot per loss
            plot_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:  # For each center used for testing
                plot_data[loss][ts_center] = {
                    f"> {volume_ranges[i][0]}, <= {volume_ranges[i][1]}": []
                    for
                    i
                    in range(len(volume_ranges))
                }
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # Lst of tst sbj pts
                for subj_path in gt_pths:
                    pred_softmax_path = os.path.join(subj_path,
                                                     f'pred_wmh_softmax_{run_name}.nii.gz')

                    pred_softmax = sitk.ReadImage(pred_softmax_path)
                    hard_pred = sitk.GetArrayFromImage(pred_softmax).argmax(
                        axis=0)

                    cc = sitk.ConnectedComponent(
                        sitk.GetImageFromArray(
                            (hard_pred > 0).astype(np.uint8)))
                    cc = sitk.RelabelComponent(cc)

                    # For each connected component, compute the volume, then, append to the
                    # corresponding list the entropy of the volume in that area.
                    # (Multiply for the voxel size to get the volume in mL)
                    for i in range(1, sitk.GetArrayFromImage(cc).max() + 1):
                        cc_mask_f = sitk.GetArrayFromImage(cc == i).flatten()
                        filt = np.where(cc_mask_f == 1)[0]
                        volume = len(filt) * np.prod(pred_softmax.GetSpacing())
                        sftmx_f = \
                            sitk.GetArrayFromImage(pred_softmax).flatten()[
                                filt]
                        # Compute entropy by first slicing the softmax image with the mask
                        # and then computing the entropy of the slice
                        ent = entropy(sftmx_f)

                        for i, (lower, upper) in enumerate(volume_ranges):
                            if lower < volume <= upper:
                                plot_data[loss][ts_center][
                                    f"> {lower}, <= {upper}"].append(ent)
                                break

        # Prepare the data for plotting
        data = []
        losses = list(plot_data.keys())
        centers_test = list(plot_data[losses[0]].keys())
        ranges = list(plot_data[losses[0]][centers_test[0]].keys())

        for loss in losses:
            for center in centers_test:
                for range_label, entropies in plot_data[loss][center].items():
                    data.extend(
                        [(loss, center, range_label, ent) for ent in
                         entropies])

        df = pd.DataFrame(data,
                          columns=['Loss', 'Center', 'Volume Range',
                                   'Entropy'])

        # Create subplots for each loss
        fig, axes = plt.subplots(1, len(losses), figsize=(16, 3), sharey=True)
        for i, loss in enumerate(losses):
            # Filter the DataFrame for the current loss
            loss_df = df[df['Loss'] == loss]

            # Create a boxplot for each center within the loss subplot
            sns.boxplot(data=loss_df, x='Volume Range', y='Entropy',
                        hue='Center',
                        ax=axes[i], palette='Set3')

            axes[i].set_title(f'Loss: {loss}')
            axes[i].set_xlabel('Volume Range')
            axes[i].set_ylabel('Entropy')

            # Manually remove the legend for all but the last subplot
            if i < len(losses) - 1:
                axes[i].get_legend().remove()

        # Set the overall title for the entire plot
        fig.suptitle(
            'Comparison of Entropy for Different Losses, Centers, and '
            f'Volume Ranges. Using {tr_center} for training',
            fontsize=16)

        # Add a single legend outside of the subplots
        handles, labels = axes[-1].get_legend_handles_labels()
        plt.legend(handles, labels, title='Center', loc='upper left',
                   bbox_to_anchor=(1, 1))

        # Adjust the layout
        plt.tight_layout()
        plt.show()


def probability_volume_ranges(plot_data,
                              volume_ranges=((0, 5), (5, 15), (15, np.inf))):
    """ Plot mean probability for each volume range

    :param plot_data: Dictionary with the required data for plotting (same
    for all plotting functions)
    :param volume_ranges: List of tuples with the volume ranges to use
    """
    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:  # Three plots: one per center used for training
        plot_data = {}  # Mean probability for each volume range
        for loss in losses:  # One subplot per loss
            plot_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:  # For each center used for testing
                plot_data[loss][ts_center] = {
                    f"> {volume_ranges[i][0]}, <= {volume_ranges[i][1]}": []
                    for
                    i
                    in range(len(volume_ranges))
                }
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # Lst of tst sbj pts
                for subj_path in gt_pths:
                    pred_softmax_path = os.path.join(subj_path,
                                                     f'pred_wmh_softmax_{run_name}.nii.gz')

                    pred_softmax = sitk.ReadImage(pred_softmax_path)
                    hard_pred = sitk.GetArrayFromImage(pred_softmax).argmax(
                        axis=0)

                    cc = sitk.ConnectedComponent(
                        sitk.GetImageFromArray((hard_pred).astype(np.uint8)))
                    cc = sitk.RelabelComponent(cc)

                    # For each connected component, compute the volume, then, append to the
                    # corresponding list the mean probability of the volume in that area.
                    # (Multiply for the voxel size to get the volume in mL)
                    for i in range(1, sitk.GetArrayFromImage(cc).max() + 1):
                        cc_mask_f = sitk.GetArrayFromImage(cc == i).flatten()
                        filt = np.where(cc_mask_f == 1)[0]
                        volume = len(filt) * np.prod(pred_softmax.GetSpacing())
                        sftmx_f = \
                            sitk.GetArrayFromImage(pred_softmax)[1].flatten()[
                                filt]

                        # Compute the mean probability
                        mean_prob = np.mean(sftmx_f)

                        for i, (lower, upper) in enumerate(volume_ranges):
                            if lower < volume <= upper:
                                plot_data[loss][ts_center][
                                    f"> {lower}, <= {upper}"].append(mean_prob)
                                break

        # Prepare the data for plotting
        data = []
        losses = list(plot_data.keys())
        centers_test = list(plot_data[losses[0]].keys())
        ranges = list(plot_data[losses[0]][centers_test[0]].keys())

        for loss in losses:
            for center in centers_test:
                for range_label, entropies in plot_data[loss][center].items():
                    data.extend(
                        [(loss, center, range_label, ent) for ent in
                         entropies])

        df = pd.DataFrame(data,
                          columns=['Loss', 'Center', 'Volume Range',
                                   'Probability'])

        # Create subplots for each loss
        fig, axes = plt.subplots(1, len(losses), figsize=(16, 3), sharey=True)
        for i, loss in enumerate(losses):
            # Filter the DataFrame for the current loss
            loss_df = df[df['Loss'] == loss]

            # Create a boxplot for each center within the loss subplot
            sns.boxplot(data=loss_df, x='Volume Range', y='Probability',
                        hue='Center', ax=axes[i], palette='Set3')

            axes[i].set_title(f'Loss: {loss}')
            axes[i].set_xlabel('Volume Range')
            axes[i].set_ylabel('Probability')

            # Manually remove the legend for all but the last subplot
            if i < len(losses) - 1:
                axes[i].get_legend().remove()

        # Set the overall title for the entire plot
        fig.suptitle(
            'Comparison of Mean Probability for Different Losses, Centers, and Volume Ranges'
            f'. Using {tr_center} for training', fontsize=16)

        # Add a single legend outside of the subplots
        handles, labels = axes[-1].get_legend_handles_labels()
        plt.legend(handles, labels, title='Center', loc='upper left',
                   bbox_to_anchor=(1, 1))

        # Adjust the layout
        plt.tight_layout()
        plt.show()


def reliability_plots(plot_data):
    num_bins = 10
    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()

    for tr_center in centers_train:  # Three plots: one per training center
        plot_data = {}  # Reliability plots
        for loss in losses:  # One subplot per loss
            plot_data[loss] = {}
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue
            for ts_center in centers_test:  # For each center used for testing
                gt_pths = get_gt_paths(
                    test_splits[ts_center])  # Lst of tst sbj pts
                preds_arr, gts_arr = [], []
                for subj_path in gt_pths:
                    gt_path = os.path.join(subj_path,
                                           f'gt_wmh_{run_name}.nii.gz')
                    pred_softmax_path = os.path.join(
                        subj_path, f'pred_wmh_softmax_{run_name}.nii.gz'
                    )

                    pred_softmax = nib.load(pred_softmax_path).get_fdata()
                    gt = nib.load(gt_path).get_fdata()
                    gt_one_hot = np.eye(2, dtype=np.uint8)[gt.astype(int)]

                    preds_arr = np.concatenate(
                        (preds_arr, pred_softmax[..., 1].flatten())
                    )
                    gts_arr = np.concatenate(
                        (gts_arr, gt_one_hot[..., 1].flatten())
                    )

                emp_probs, pred_probs = calibration_curve(gts_arr, preds_arr,
                                                          n_bins=num_bins)
                plot_data[loss][ts_center] = {
                    'emp_probs': emp_probs,
                    'pred_probs': pred_probs,
                    'ece': get_ece(preds_arr, gts_arr.astype(int))
                }

        fig, axes = plt.subplots(1, len(centers_test), figsize=(13.5, 4.5),
                                 sharey=True)

        [axes[i].plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated') for i
         in range(len(centers_test))]  # Add the perfect calibration line
        for loss, ts_center_data in plot_data.items():
            for i, (ts_center, data) in enumerate(ts_center_data.items()):
                ax = axes[i]

                # Extract data from plot_data
                emp_probs = data['emp_probs']
                pred_probs = data['pred_probs']
                ece = data['ece']

                ax.plot(pred_probs, emp_probs, 'o-',
                        label=f'{loss} - ECE: {ece:.4e}')
                ax.set_xlabel('Predicted probability')
                ax.set_ylabel('Empirical probability')
                ax.set_ylim([-0.05, 1.05])
                ax.legend()

                # Set titles, labels, and other formatting
                ax.set_title(f'Center: {ts_center}')
                ax.set_xlabel('Empirical probability')
                ax.set_ylabel('Predicted probability')

                # Make the subplot square
                ax.set_aspect('equal', 'box')

        for ax in axes[1:]:
            ax.set_ylabel('')

        plt.suptitle(f'Reliability plots using {tr_center} for training')

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice


def dice_norm_metric(y_true, y_pred, r=0.001):
    """
    Compute Normalised Dice Coefficient (nDSC),
    False positive rate (FPR),
    False negative rate (FNR) for a single example.

    :param y_true: Ground truth segmentation
    :param y_pred: Predicted segmentation
    :param r: Normalization constant

    :return: nDSC, FPR, FNR
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = y_true.astype("float32")
    seg = y_pred.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm


def error_retention_curve(plot_data, metric='nDSC'):
    """
    Compute error retention curve and R-AUC using the Dice coefficient as the error metric.

    :param plot_data: Dictionary with the required data for plotting
    :param metric: Metric to use for computing the error retention curve.
    Can be either 'nDSC' or 'Dice'.
    """

    centers_train, runs_to_compare, centers_test, test_splits, losses = plot_data.values()
    metric_fn = dice_norm_metric if metric == 'nDSC' else dice_score

    retention_curves = {}
    for tr_center in centers_train:
        for loss in losses:
            run_name = runs_to_compare[f'{loss} {tr_center}']
            if f'{loss} {tr_center}' not in runs_to_compare.keys():
                print(f'No run for {run_name}.')
                continue

            for ts_center in centers_test:
                gt_pths = get_gt_paths(test_splits[ts_center])
                retention_curves_arr = []

                for subj_path in gt_pths:
                    gt_path = os.path.join(subj_path,
                                           f'gt_wmh_{run_name}.nii.gz')
                    pred_softmax_path = os.path.join(subj_path,
                                                     f'pred_wmh_softmax_{run_name}.nii.gz')
                    pred_hard_path = os.path.join(subj_path,
                                                  f'pred_wmh_hard_{run_name}.nii.gz')

                    gt = nib.load(gt_path).get_fdata()
                    pred_softmax = nib.load(pred_softmax_path).get_fdata()
                    pred_hard = nib.load(pred_hard_path).get_fdata()

                    pos_class = pred_softmax[:, :, :, 1].flatten()
                    pred_hard_class = pred_hard.flatten()
                    gt_class = gt.flatten()

                    uncertainties = entropy(pos_class, apply_mean=False)

                    # Initialize retention data for this iteration
                    retention_curve = []

                    # Sample by fractions of samples
                    for frac in np.linspace(1.0, 0.0, 11):
                        num_samples = int(frac * len(uncertainties))
                        sorted_indices = np.argsort(uncertainties)
                        retained_indices = sorted_indices[:num_samples]

                        replaced_pred = pred_hard_class.copy()
                        replaced_pred[retained_indices] = gt_class[
                            retained_indices]

                        # Compute Dice and add to retention curve
                        dice = metric_fn(gt_class, replaced_pred)
                        retention_curve.append(dice)
                    retention_curves_arr.append(retention_curve)
                retention_curves[f'{loss} {tr_center} {ts_center}'] = np.mean(
                    retention_curves_arr, axis=0)

    subplot_size = (4, 4)
    for tr_center in centers_train:
        num_subplots = len(centers_test)
        fig_width = (num_subplots + 2) * subplot_size[0]
        plt.figure(figsize=(fig_width, subplot_size[1]))
        plt.suptitle(f'Training Center: {tr_center}')

        for i, ts_center in enumerate(centers_test, start=1):
            plt.subplot(1, num_subplots, i,
                        aspect='equal')  # Set aspect to 'equal'
            plt.title(f'Test Center: {ts_center}')

            for loss in losses:
                key = f'{loss} {tr_center} {ts_center}'
                if key in retention_curves:
                    curve = retention_curves[key]
                    auc_value = auc(np.linspace(1, 0, 11), curve[
                                                           ::-1])  # Reverse the curve for correct AUC
                    plt.plot(np.linspace(1, 0, 11), curve[::-1],
                             # Reverse the curve for correct plotting
                             label=f'{loss} (AUC={auc_value:.2f})')

            plt.xlabel('Fraction of Retained Samples')
            plt.ylabel('Mean Dice Coefficient')
            plt.legend()

            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)

        plt.subplots_adjust(wspace=0.3)  # Adjust the value of wspace as needed
        plt.show()
