import cv2
import numpy as np
import torch


class SegmentationModel:
    """
    Binary segmentation model for grayscale (1-channel) images. Provides inference primitives.

    Args:
        model_path (str): path to model weights
        activation (callable): torch activation function
    """
    def __init__(self, model_path, activation=torch.sigmoid):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.activation = activation

    def _binary_activation(self, logits, binarization_threshold=0.5):
        return (self.activation(logits) >= binarization_threshold).int()

    def _to_tensor(self, numpy_image):
        numpy_image = numpy_image[np.newaxis, :, :].astype("float32")  # (num of channels) dimension must be provided
        tensor = torch.from_numpy(numpy_image).to(self.device)
        return tensor.unsqueeze(0)  # batch size dimension must be provided to the network

    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Make sure image dimensions can be divided by 32
        normalized_image = (image - image.min()) / (image.max() - image.min())
        return self._to_tensor(normalized_image)

    def predict(self, image, is_image_from_path=True, flip_tta=False, binarization=True):  # image can be either an image path or a nomralized image array
        if is_image_from_path:
            image = self._load_image(image)
        else:
            image = self._to_tensor(image)
        if flip_tta: # whetever to predict segmentation mask on both the original and the flipped image and then average them
            unflipped_mask = self.model.predict(image)  # switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
            flipped_image = torch.flip(image, [-1])
            flipped_mask = self.model.predict(flipped_image)
            mask = (unflipped_mask + torch.flip(flipped_mask, [-1])) / 2 # mask probability averaging
        else:
            mask = self.model.predict(image)  # switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        if binarization:  # whetever to apply binarization thresholding after activation
            binary_mask = self._binary_activation(mask)
            return binary_mask.squeeze().cpu().numpy()
        else:
            return self.activation(mask).squeeze().cpu().numpy()
