import os
import random

import lightning as L
import torchio as tio
from torch.utils.data import DataLoader

from src.datamodules.transforms import get_preprocessing

TRAINING = ['tr', 'train', 'training']
VALIDATION = ['v', 'val', 'validation']
TEST = ['te', 'test', 'testing']


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


class WMHDataModule(L.LightningDataModule):
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
    :param tio_num_workers: Number of workers for TorchIO
    :param samples_per_volume: Number of samples per volume
    :param queue_length: Queue length
    :param predict_split: Split to use for prediction
    """

    def __init__(self, data_dir: str, batch_size: int, centers: str,
                 split_ratios: list, patch_size: int = None, seed: int = 42,
                 tio_num_workers: int = None, samples_per_volume: int = None,
                 queue_length: int = None, predict_split: str = 'test'):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)  # Data directory
        self.batch_size = batch_size  # Batch size
        self.centers = centers  # Centers to use
        self.train_ds = None  # Training dataset
        self.split_ratios = split_ratios  # Ratios for training/validation/test
        self.seed = seed  # Random seed
        self.val_ds = None  # Validation dataset
        self.test_ds = None  # Test dataset
        self.subjects = None  # List of subjects
        self.transforms = None  # Transforms to apply to the data
        self.patch_size = (patch_size, patch_size, patch_size)  # Patch size
        self.samples_per_volume = samples_per_volume  # Samples per volume
        self.queue_length = queue_length  # Queue length
        self.tio_num_workers = tio_num_workers  # TorchIO workers
        self.ids = None
        self.label_name = 'wmh'
        self.label_probs = {0: 0.1, 1: 0.9}
        self.centers_dict = None  # Dictionary with the centers
        self.subj_train = None  # List of subjects for training
        self.subj_val = None  # List of subjects for validation
        self.subj_test = None  # List of subjects for testing
        self.predict_split = predict_split  # Split to use for prediction

    def get_centers_dict(self):
        """  Convert the centers string to a dictionary

        Example: "phase1:cent1;phase2:cent2" -> {"phase1": "cent1",
        "phase2": "cent2"}

        :return: Dictionary with the centers
        """
        self.centers_dict = {
            k: v.split(',') for k, v in
            [c.split(':') for c in self.centers.split(';')]
        }
        return self.centers_dict

    def get_expl_folders(self):
        """ Get the list of folders with the data from the centers dictionary

        Use the centers dictionary to get the list of folders with the data. The
         folders are returned in a list.

        :return: List of folders with the data for each split/center.
        """
        expl_folders = []
        for split, ctrs in self.centers_dict.items():
            for ctr in ctrs:
                ctr_path = os.path.join(self.data_dir, split, ctr)
                if ctr == 'Amsterdam':  # It has 3 sub-folders
                    for f in os.listdir(ctr_path):
                        expl_folders.append(os.path.join(ctr_path, f))
                else:
                    expl_folders.append(ctr_path)
        return expl_folders

    def split_aux(self, aux_train_imgs, tr_im, val_im, tst_im):
        """ Split the list of images into train, validation, and test sets

        Using the split ratios, split the list of images into train, validation,
        and test sets. The sets are returned in the lists passed by reference.
        For reproducibility, the random seed is set before shuffling the list.

        :param aux_train_imgs: List of images to split
        :param tr_im: Train set
        :param val_im: Validation set
        :param tst_im: Test set
        :return: None
        """
        random.seed(self.seed)

        # Calculate the number of images for each set based on the splits
        n_train = int(len(aux_train_imgs) * self.split_ratios[0])
        n_val = int(len(aux_train_imgs) * self.split_ratios[1])

        # Shuffle the list of images randomly
        random.shuffle(aux_train_imgs)

        # Split the shuffled list into train, validation, and test sets
        train_set = aux_train_imgs[:n_train]
        val_set = aux_train_imgs[n_train:n_train + n_val]
        test_set = aux_train_imgs[n_train + n_val:]

        # Assign the sets to the output lists passed by reference
        tr_im.extend(train_set)
        val_im.extend(val_set)
        tst_im.extend(test_set)

    def generate_splits(self):
        """ Generate the splits for training, validation, and test sets

        Generate the splits for training, validation, and test sets. The splits
        are generated using the centers dictionary.
        Each split consists in a list of lists, where each list contains the
        paths to the T1, FLAIR, and WMH images for a subject.

        :return: Tuple with the train, validation, and test sets, respectively.
        """
        if self.centers_dict is None:
            self.get_centers_dict()
        expl_folders = self.get_expl_folders()

        tr_im, val_im, tst_im = [], [], []

        for spl_ctr_fld in expl_folders:
            aux_train_imgs = []
            for subj in os.listdir(spl_ctr_fld):
                subj_path = os.path.join(spl_ctr_fld, subj)
                if 'training' in spl_ctr_fld.split(os.path.sep)[-3:-1]:
                    aux_train_imgs.append([
                        os.path.join(subj_path, 'pre', 'T1.nii.gz'),
                        os.path.join(subj_path, 'pre', 'FLAIR.nii.gz'),
                        os.path.join(subj_path, 'wmh.nii.gz')
                    ])
                else:
                    tst_im.append([
                        os.path.join(subj_path, 'pre', 'T1.nii.gz'),
                        os.path.join(subj_path, 'pre', 'FLAIR.nii.gz')
                    ])
            self.split_aux(aux_train_imgs, tr_im, val_im, tst_im)

        if self.predict_split:  # Allow to predict on train/val/test splits
            tst_im = tr_im if self.predict_split in TRAINING \
                else val_im if self.predict_split in VALIDATION \
                else tst_im  # Default case (do nothing)

        return tr_im, val_im, tst_im

    def create_subjects(self, split):
        """ Create the subjects for the TorchIO dataset

        From a list of lists with the paths to the images and labels, create the
        subjects for the TorchIO dataset.

        :param split: List of lists with the paths to the images and labels. It
        corresponds to a split (train, validation, or test).
        :return subjects: List of subjects for the TorchIO dataset
        """
        subjects = []
        for im in split:
            if len(im) == 3:
                subject = MySubject(
                    t1=tio.ScalarImage(im[0]),
                    flair=tio.ScalarImage(im[1]),
                    wmh=tio.LabelMap(im[2])
                )
            else:
                subject = MySubject(
                    t1=tio.ScalarImage(im[0]),
                    flair=tio.ScalarImage(im[1])
                )
            subjects.append(subject)
        return subjects

    def prepare_data(self):
        """ Prepare the data

        Get the paths to the images and labels and create the TorchIO dataset.
        """
        tr_sp, vl_sp, ts_sp = self.generate_splits()
        self.subj_train = self.create_subjects(tr_sp)
        self.subj_val = self.create_subjects(vl_sp)
        self.subj_test = self.create_subjects(ts_sp)

    def setup(self, stage: str):
        # Some subjects have also 2 as "other pathology". We remap it to 0
        self.transforms = get_preprocessing(include_labels=True)

        if stage == "fit":
            self.train_ds = tio.SubjectsDataset(
                self.samples_per_volume * self.subj_train, self.transforms
            )
            self.val_ds = tio.SubjectsDataset(
                self.samples_per_volume * self.subj_val, self.transforms
            )
        if stage == "test":
            if self.split_ratios[2] == 0:
                print(f'W: Test split ratio is 0. Will predict on all data '
                      f'({self.centers}).')
                self.subj_test = self.subj_train + self.subj_val + \
                                 self.subj_test

            self.test_ds = tio.SubjectsDataset(self.subj_test,
                                               self.transforms)

    def get_dataloader(self, dataset, test=False):
        """ Get the dataloader for the dataset

        Get the dataloader for the dataset. If the patch size is not specified,
        the dataloader will return full volume images. Otherwise, it will return
        patches of the specified size.

        For the test set, the dataloader will return full volume images always.

        :param dataset: Dataset split (train, validation, or test).
        :param test: Boolean indicating if the dataset is the test set (won't
        return patches).
        :return: Dataloader for the corresponding dataset.
        """
        if not any(self.patch_size) or test:
            return DataLoader(dataset, self.batch_size)  # Full volume
        else:
            sampler = tio.data.LabelSampler(self.patch_size,
                                            self.label_name,
                                            self.label_probs)
            patches_queue = tio.Queue(
                dataset,
                self.queue_length,
                self.samples_per_volume,
                sampler,
                num_workers=self.tio_num_workers,
            )
            return DataLoader(patches_queue, self.batch_size)

    def train_dataloader(self):
        return self.get_dataloader(self.train_ds)

    def val_dataloader(self):
        return self.get_dataloader(self.val_ds)

    def test_dataloader(self):
        return self.get_dataloader(self.test_ds, test=True)
