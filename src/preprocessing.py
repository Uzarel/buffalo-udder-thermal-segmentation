import albumentations as albu
import numpy as np


def normalize_image(image, **kwargs):
    return (image - image.min()) / (image.max() - image.min())


def binarize_mask(mask, **kwargs):
    return mask // 255  # (0 -> 0, 255 -> 1)


def to_tensor(x, **kwargs):
    return x[np.newaxis, :, :].astype(
        "float32"
    )  # (num of channels) dimension must be provided


def get_preprocessing():
    return albu.Compose(
        [
            albu.Lambda(image=normalize_image, mask=binarize_mask),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    )
