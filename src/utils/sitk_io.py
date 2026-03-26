from __future__ import annotations

import SimpleITK as sitk
import torch


def restore_metadata_as_sitk(
    img: torch.Tensor,
    source_img: str | sitk.Image,
) -> sitk.Image:
    """Restore the metadata from the source image

    Given an image, and a source image path, load the source image and copy the
    metadata to the images in the dictionary.

    In case the img is padded, it'll be cropped according to the source image
    shape.

    :param img: Image as ndarray
    :param source_img: Reference image path
    :return: Image with restored metadata as SimpleITK image
    """
    shape = img.shape
    source_img = sitk.ReadImage(source_img) if isinstance(source_img, str) else source_img
    if len(shape) == 4:
        img_slices = []
        for i in range(shape[0]):
            img_slices.append(restore_metadata_as_sitk(img[i], source_img))
        img = sitk.JoinSeries(img_slices)
    elif len(shape) == 3:
        img = sitk.GetImageFromArray(img)
        img = sitk.PermuteAxes(img, (2, 1, 0))
        img = sitk.Flip(img, [True, True, False])  # flip the first axis
        if img.GetSize() != source_img.GetSize():
            sz = source_img.GetSize()
            img = img[0 : sz[0], 0 : sz[1], 0 : sz[2]]
        img.CopyInformation(source_img)
    else:
        raise ValueError("Image dimension not supported")
    return img
