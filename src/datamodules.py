import os

import pytorch_lightning as pl
import torchio as tio
from torch.utils.data import random_split, DataLoader


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
    :param train_ratio: Ratio of training data to use
    :param seed: Random seed
    """

    def __init__(self, data_dir: str, batch_size: int, centers: str,
                 train_ratio: float, patch_size: int, seed: int):
        super().__init__()
        self.data_dir = data_dir  # Path to the dataset
        self.batch_size = batch_size  # Batch size
        self.centers = centers  # Centers to use
        self.train_dataset = None  # Training dataset
        self.train_ratio = train_ratio  # Ratio of training data to use
        self.seed = seed  # Random seed
        self.val_dataset = None  # Validation dataset
        self.test_dataset = None  # Test dataset
        self.subjects = None  # List of subjects
        self.transforms = None  # Transforms to apply to the data
        self.patch_size = (patch_size, patch_size, patch_size)  # Patch size

    def prepare_data(self):
        """ Prepare the data

        Get the paths to the images and labels and create the TorchIO dataset.
        """
        paths_lists = get_paths_wmh(self.data_dir, self.centers)
        self.subjects = []
        for paths_list in paths_lists:
            if len(paths_list) < 3:
                subject = tio.Subject(
                    t1=tio.ScalarImage(paths_list[0]),
                    flair=tio.LabelMap(paths_list[1])
                )
            else:
                subject = tio.Subject(
                    t1=tio.ScalarImage(paths_list[0]),
                    flair=tio.ScalarImage(paths_list[1]),
                    wmh=tio.LabelMap(paths_list[2])
                )
            self.subjects.append(subject)

    def setup(self, stage: str):
        images = self.subjects
        if stage == "fit" or stage is None:
            n_train = int(self.train_ratio * len(images))
            n_val = len(images) - n_train
            train_images, val_images = random_split(images, [n_train, n_val])

            # Some subjects have also 2 as "other pathology". We remap it to 0
            remapping = {2: 0}
            remap = tio.RemapLabels(remapping)
            transform = tio.Lambda(
                lambda x: remap(x) if x['type'] == 'LabelMap' else x
            )  # Apply just to LabelMaps
            self.transforms = tio.Compose([transform])

            self.train_dataset = tio.SubjectsDataset(train_images)
            self.val_dataset = tio.SubjectsDataset(val_images)

        if stage == "test" or stage is None:
            self.test_dataset = tio.SubjectsDataset(images)

    def train_dataloader(self):
        train_sampler = tio.data.UniformSampler(patch_size=(64, 64, 64))
        return DataLoader(self.train_dataset, self.batch_size,
                          sampler=train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size)

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

    images = []
    expl_folders = []
    for split, ctrs in centers.items():
        for ctr in ctrs:
            ctr_path = os.path.join(src_path, ctr)
            if ctr == 'Amsterdam':  # It has 3 subfolders
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
