import configparser
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import sum
from tqdm import tqdm

from src.data import Dataset
from src.model import SegmentationModel
from src.preprocessing import get_preprocessing


# Config file parser
config = configparser.ConfigParser()
config.read('config.ini')

# Constants
DEVICE = config.get('EVAL', 'DEVICE')
BATCH_SIZE = int(config.get('EVAL', 'BATCH_SIZE'))
MODEL_PATH = config.get('EVAL', 'MODEL_PATH')
X_TEST_PATH = config.get('DATA_PATHS', 'X_TEST')
Y_TEST_PATH = config.get('DATA_PATHS', 'Y_TEST')
VAL_LOGS_PATH = config.get('EVAL', 'VAL_LOGS_PATH')


# Better ticks selector
def select_ticks(ticks, sorting="descending", threshold=0.02):
    assert sorting in ["ascending", "descending"]
    if sorting == "ascending":
        sorted_ticks = sorted(ticks)
        selected_ticks = [sorted_ticks[0]]
        for sorted_tick in sorted_ticks[1:]:
            if sorted_tick - selected_ticks[-1] > threshold:
                selected_ticks.append(sorted_tick)
    else:  # sorting == 'descending'
        sorted_ticks = sorted(ticks, reverse=True)
        selected_ticks = [sorted_ticks[0]]
        for sorted_tick in sorted_ticks[1:]:
            if selected_ticks[-1] - sorted_tick > threshold:
                selected_ticks.append(sorted_tick)
    return selected_ticks


# Loss-mIoU Plot
def plot_loss_miou(val_logs_path):
    df = pd.read_csv(val_logs_path)

    joint_loss = df["joint_loss"]
    iou_score = df["iou_score"]

    fig, ax1 = plt.subplots(figsize=(18, 9))

    epochs = range(1, len(joint_loss) + 1)

    ax1.plot(epochs, joint_loss, label="Joint Loss", marker="o", color="tab:blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Joint Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(select_ticks(epochs, sorting="ascending", threshold=2))
    ax1.set_yticks(select_ticks(joint_loss, sorting="ascending"))

    ax2 = ax1.twinx()
    ax2.plot(epochs, iou_score, label="IoU Score", marker="x", color="tab:orange")
    ax2.set_ylabel("IoU Score", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_yticks(select_ticks(iou_score, sorting="descending"))
    best_idx = iou_score.idxmax()
    ax2.axvline(x=best_idx + 1, color="red", linestyle="--", label="Max IoU Score")

    fig.suptitle("Joint (Soft-BCE + Dice + Focal) Loss/mIoU Score-Epoch Curve")
    fig.legend(loc="upper right")
    ax1.grid(visible=True, color="tab:blue", linestyle="-", alpha=0.5)
    ax2.grid(visible=True, color="tab:orange", linestyle="-", alpha=0.5)
    plt.savefig('out/plot_loss_miou.png')
    plt.close()


# Pixel-level Precision-Recall-F1 Plot
def plot_precision_recall(val_logs_path):
    df = pd.read_csv(val_logs_path)

    precision = df["precision"]
    recall = df["recall"]
    fscore = df["fscore"]
    epochs = range(1, len(precision) + 1)

    plt.figure(figsize=(18, 9))

    plt.plot(epochs, precision, label="Precision", marker="o", color="tab:blue")
    plt.plot(epochs, recall, label="Recall", marker="x", color="tab:orange")
    plt.plot(epochs, fscore, label="F1-Score", marker="d", color="tab:green")
    best_idx = fscore.idxmax()
    plt.axvline(x=best_idx + 1, color="tab:red", linestyle="--", label="Max F1-score")

    plt.title("Precision/Recall-Epoch Curve")
    plt.legend(loc="lower right")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.xticks(select_ticks(epochs, sorting="ascending", threshold=2))
    plt.yticks(select_ticks(pd.concat([precision, recall, fscore])))
    plt.grid(visible=True, color="tab:purple", linestyle="-", alpha=0.5)
    plt.savefig('out/plot_precision_recall.png')
    plt.close()


# Accuracy
def good_prediction(gt_tensor, pr_tensor, iou_threshold=0.5, eps=1e-7):
    intersection = sum(gt_tensor * pr_tensor)
    union = sum(gt_tensor) + sum(pr_tensor) - intersection + eps
    iou_scores = (intersection + eps) / union
    return 1 if iou_scores > iou_threshold else 0

def compute_accuracies(dataloader, segmentation_model, start_threshold, end_threshold, step):
    predictions = dict()
    for image_batch, gt_mask_batch in tqdm(dataloader, desc="Computing prediction masks"):
        image_batch, gt_mask_batch = image_batch.to(DEVICE), gt_mask_batch.to(DEVICE)
        pr_mask_batch = segmentation_model.model.predict(image_batch)
        pr_mask_batch = segmentation_model._binary_activation(pr_mask_batch)
        for iou_threshold in np.arange(start_threshold, end_threshold, step):
            prediction = good_prediction(gt_mask_batch, pr_mask_batch, iou_threshold)
            if iou_threshold not in predictions.keys():
                predictions[iou_threshold] = list([prediction])
            else:
                predictions[iou_threshold].append(prediction)
    results = {iou_threshold: np.mean(preds) for iou_threshold, preds in predictions.items()}
    return list(results.keys()), list(results.values())

# Accuracy-IoU threshold plot
def plot_accuracies(iou_thresholds, accuracies):
    plt.figure(figsize=(18, 12))  # Create a new figure
    plt.plot(iou_thresholds, accuracies, marker="o", label="Accuracy")
    plt.xlabel("IoU")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs IoU Threshold")
    plt.xticks(iou_thresholds)
    plt.yticks(select_ticks(accuracies, sorting='descending', threshold=0.03))
    plt.grid(color='tab:blue', alpha=0.5)
    plt.savefig('out/plot_accuracies.png')
    plt.close()  # Close the figure to avoid overlap


if __name__ == "__main__":
    test_dataset = Dataset(X_TEST_PATH, Y_TEST_PATH, preprocessing=get_preprocessing())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    plot_loss_miou(VAL_LOGS_PATH)
    plot_precision_recall(VAL_LOGS_PATH)

    model = SegmentationModel(MODEL_PATH)
    iou_thresholds, accuracies = compute_accuracies(test_loader, model, 0.5, 1.0, 0.02)
    plot_accuracies(iou_thresholds, accuracies)
