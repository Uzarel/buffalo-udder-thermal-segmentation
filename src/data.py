import cv2
import os

from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """
    Read images and binary masks, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        # Images and masks share the same filename
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        # Conversion to a binary mask (0 -> 0, 255 -> 1)
        mask = mask // 255
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        return image, mask

    def __len__(self):
        return len(self.ids)
