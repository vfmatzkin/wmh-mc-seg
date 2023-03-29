import os

import pytorch_lightning as pl
import torchio as tio
from torch import Generator
from torch.utils.data import random_split, DataLoader


class MySubject(tio.Subject):
    """ MySubject class

    If the tolerance value is too small and the different images from the same
    subject have slightly different values, TorchIO won't accept the images.
    This happened with T1/FLAIR/WMH images from the WMH dataset.
    """

    def check_consistent_attribute(self, *args, **kwargs) -> None:
        kwargs['relative_tolerance'] = 1e-5
        kwargs['absolute_tolerance'] = 1e-5
        return super().check_consistent_attribute(*args, **kwargs)


class WMHDataModule(pl.LightningDataModule):
    """ WMHDataModule

    This class is used to load the WMH dataset and prepare the data for
    training/testing.
    Note that this class should be instantiated separately for training and
    testing, since centers indicate the centers used for just one of the
    phases only.

    In case of training, the data is split into training and validation
    following the train_ratio parameter.

    :param data_dir: Path to the dataset
    :param batch_size: Batch size
    :param centers: Centers to use. The format is: "phase1:cent1;phase2:cent2"
    :param split_ratios: Ratio of training data to use
    :param seed: Random seed
    """

    def __init__(self, data_dir: str, batch_size: int, centers: str,
                 split_ratios: list, patch_size: int, seed: int,
                 samples_per_volume: int, queue_length: int,
                 tio_num_workers: int):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)  # Data directory
        self.batch_size = batch_size  # Batch size
        self.centers = centers  # Centers to use
        self.train_dataset = None  # Training dataset
        self.split_ratios = split_ratios  # Ratios for training/validation/test
        self.seed = seed  # Random seed
        self.val_dataset = None  # Validation dataset
        self.test_dataset = None  # Test dataset
        self.subjects = None  # List of subjects
        self.transforms = None  # Transforms to apply to the data
        self.patch_size = (patch_size, patch_size, patch_size)  # Patch size
        self.samples_per_volume = samples_per_volume  # Samples per volume
        self.queue_length = queue_length  # Queue length
        self.tio_num_workers = tio_num_workers  # TorchIO workers

    def prepare_data(self):
        """ Prepare the data

        Get the paths to the images and labels and create the TorchIO dataset.
        """
        paths_lists = get_paths_wmh(self.data_dir, self.centers)
        self.subjects = []
        for paths_list in paths_lists:
            if len(paths_list) < 3:
                subject = MySubject(
                    t1=tio.ScalarImage(paths_list[0]),
                    flair=tio.LabelMap(paths_list[1])
                )
            else:
                subject = MySubject(
                    t1=tio.ScalarImage(paths_list[0]),
                    flair=tio.ScalarImage(paths_list[1]),
                    wmh=tio.LabelMap(paths_list[2])
                )
            self.subjects.append(subject)

    def setup(self, stage: str):
        images = self.subjects
        n = len(images)
        p_tr, p_va, p_te = self.split_ratios

        n_train = int(p_tr * n)
        n_val = int(p_va * n) if p_te != 0 else n - n_train
        n_test = n - n_train - n_val
        train_images, val_images, test_images = random_split(
            images, [n_train, n_val, n_test],
            generator=Generator().manual_seed(self.seed)
        )
        train_images = train_images.dataset
        val_images = val_images.dataset
        test_images = test_images.dataset

        if stage == "fit" or stage is None:

            # Some subjects have also 2 as "other pathology". We remap it to 0
            self.transforms = tio.Compose([
                tio.ZNormalization(include=['t1', 'flair']),
                tio.ToCanonical(),
                tio.Resample('t1'),
                tio.RemapLabels({2: 0}, include=['wmh']),
                tio.OneHot(include=['wmh']),
            ])

            self.train_dataset = tio.SubjectsDataset(
                self.samples_per_volume * train_images, self.transforms
            )
            self.val_dataset = tio.SubjectsDataset(
                self.samples_per_volume * val_images, self.transforms
            )
        if stage == "test" or stage is None:
            if self.split_ratios[2] == 0:
                self.test_dataset = tio.SubjectsDataset(images)
            else:
                self.test_dataset = tio.SubjectsDataset(test_images)
                
    def train_dataloader(self):
        train_sampler = tio.data.LabelSampler(self.patch_size,
                                              label_name='wmh',
                                              label_probabilities={0: 0.5,
                                                                   1: 0.5})
        patches_queue = tio.Queue(
            self.train_dataset,
            self.queue_length,
            self.samples_per_volume,
            train_sampler,
            num_workers=self.tio_num_workers,
        )
        return DataLoader(patches_queue, self.batch_size)

    def val_dataloader(self):
        val_sampler = tio.data.LabelSampler(self.patch_size,
                                            label_name='wmh',
                                            label_probabilities={0: 0.5,
                                                                 1: 0.5})
        patches_queue = tio.Queue(
            self.val_dataset,
            self.queue_length,
            self.samples_per_volume,
            val_sampler,
            num_workers=self.tio_num_workers,
        )
        return DataLoader(patches_queue, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size)


def get_paths_wmh(src_path: str, centers: str):
    """ Get images and labels paths from a folder.

    example parameters:

    data_path = os.path.expanduser("~/Code/datasets/wmh/")
    centers = 'training:Utrecht,Amsterdam;test:Utrecht'

    This can be used, for example, for getting predictions on both training and
    test sets centers.

    This will find all the T1/FLAIR images in the 'pre' folder of Amsterdam
    folder (for all scanners and subjects of the training set) and the Utrecht
    test images (for the only scanner and all subjects).
    For more details, see the documentation: https://wmh.isi.uu.nl/methods/example-python/  # noqa: E501


    :param src_path: Path of the extracted dataset.
    :param centers: List of centers to use.
    :return: List of images and labels paths.
    """
    centers = {
        k: v.split(',') for k, v in [c.split(':') for c in centers.split(';')]
    }

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Source folder {src_path} does not exist. The "
            f"dataset should be placed in this folder.")

    images = []
    expl_folders = []
    for split, ctrs in centers.items():
        for ctr in ctrs:
            ctr_path = os.path.join(src_path, split, ctr)
            if ctr == 'Amsterdam':  # It has 3 sub-folders
                for f in os.listdir(ctr_path):
                    expl_folders.append(os.path.join(ctr_path, f))
            else:
                expl_folders.append(ctr_path)

    for folder in expl_folders:
        for subj in os.listdir(folder):
            subj_path = os.path.join(folder, subj)
            if 'training' in folder.split(os.path.sep):
                images.append([
                    os.path.join(subj_path, 'pre', 'T1.nii.gz'),
                    os.path.join(subj_path, 'pre', 'FLAIR.nii.gz'),
                    os.path.join(subj_path, 'wmh.nii.gz')
                ])
            else:
                images.append([
                    os.path.join(subj_path, 'pre', 'T1.nii.gz'),
                    os.path.join(subj_path, 'pre', 'FLAIR.nii.gz')
                ])

    return images
