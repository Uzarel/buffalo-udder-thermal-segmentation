import albumentations as albu
import numpy as np


def to_tensor(x, **kwargs): # TODO: This is the computational bottleneck (floats instead of integers)
    return x[np.newaxis, :, :].astype("float32") # batch size dimension must be provided to the network

def get_preprocessing():
    return albu.Lambda(image=to_tensor, mask=to_tensor)
