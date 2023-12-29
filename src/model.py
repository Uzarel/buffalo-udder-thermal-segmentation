import cv2
import numpy as np
import torch


class SegmentationModel:
    """
    Binary segmentation model for grayscale (1-channel) images. Provides inference primitives.

    Args:
        model_path (str): path to model weights
    """
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)

    @staticmethod
    def binary_activation(logits, activation=torch.sigmoid, binarization_threshold=0.5):
        return (activation(logits) >= binarization_threshold).int()

    def _to_tensor(self, numpy_image):
        tensor = torch.from_numpy(numpy_image).to(self.device)
        return tensor.unsqueeze(0)  # batch size dimension must be provided to the network

    def _load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype("float32")  # Make sure image dimensions can be divided by 32
        image = image[np.newaxis, :, :]  # (num of channels) dimension must be provided
        return self._to_tensor(image)

    def predict(self, image_path):
        image = self._load_image(image_path)
        pr_mask = self.model.predict(image)  # Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        binary_mask = self.binary_activation(pr_mask, activation=torch.sigmoid)
        return binary_mask.squeeze().cpu()
