import albumentations as albu
import numpy as np


def to_tensor(x, **kwargs): # TODO: This is the computational bottleneck (floats instead of integers)
    return x[np.newaxis, :, :].astype("float32") # (num of channels) dimension must be provided

def get_preprocessing():
    return albu.Lambda(image=to_tensor, mask=to_tensor)
