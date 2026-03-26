import torchio as tio


def get_preprocessing(include_labels=True):
    """Return a tio.Compose preprocessing pipeline.

    :param include_labels: If True, include label-specific transforms
        (RemapLabels, OneHot). Set to False for inference without ground truth.
    :return: tio.Compose pipeline
    """
    ops = [
        tio.ZNormalization(include=['t1', 'flair']),
        tio.ToCanonical(),
        tio.Resample('t1'),
        tio.EnsureShapeMultiple(2 ** 4),
    ]
    if include_labels:
        # Insert before EnsureShapeMultiple (second-to-last position)
        ops.insert(-1, tio.RemapLabels({2: 0}, include=['wmh']))
        ops.append(tio.OneHot(include=['wmh']))
    return tio.Compose(ops)
