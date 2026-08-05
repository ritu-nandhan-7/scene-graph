from __future__ import annotations


def build_transforms(
    preprocess,
):
    """
    Build train and validation transforms.
    """

    train_transform = preprocess

    val_transform = preprocess

    return train_transform, val_transform