import importlib.util
from pathlib import Path
import sys

import torchio as tio

# Import transforms directly to avoid src/datamodules/__init__.py pulling in
# WMHDataModule → lightning (a heavy dep not needed for this test).
_transforms_path = Path(__file__).parent.parent / 'src' / 'datamodules' / 'transforms.py'
_spec = importlib.util.spec_from_file_location('datamodules.transforms', _transforms_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['datamodules.transforms'] = _mod
_spec.loader.exec_module(_mod)
get_preprocessing = _mod.get_preprocessing


def test_get_preprocessing_returns_compose():
    pipeline = get_preprocessing(include_labels=True)
    assert isinstance(pipeline, tio.Compose)


def test_get_preprocessing_without_labels_returns_compose():
    pipeline = get_preprocessing(include_labels=False)
    assert isinstance(pipeline, tio.Compose)


def test_get_preprocessing_with_labels_includes_remap_and_onehot():
    pipeline = get_preprocessing(include_labels=True)
    transform_types = [type(t) for t in pipeline.transforms]
    assert tio.RemapLabels in transform_types
    assert tio.OneHot in transform_types


def test_get_preprocessing_without_labels_excludes_remap_and_onehot():
    pipeline = get_preprocessing(include_labels=False)
    transform_types = [type(t) for t in pipeline.transforms]
    assert tio.RemapLabels not in transform_types
    assert tio.OneHot not in transform_types


def test_get_preprocessing_with_labels_has_more_transforms():
    with_labels = get_preprocessing(include_labels=True)
    without_labels = get_preprocessing(include_labels=False)
    assert len(with_labels.transforms) > len(without_labels.transforms)
