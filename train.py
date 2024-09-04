import albumentations as albu
import configparser
import os
import pandas as pd
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from torch import load, save
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.callbacks import EarlyStopper
from src.data import Dataset
from src.losses import JointLoss
from src.misc import load_logs
from src.preprocessing import get_preprocessing


# Config file parser
config = configparser.ConfigParser()
config.read('config.ini')

# Constants
HEIGHT = int(config.get('IMAGE', 'HEIGHT'))
WIDTH = int(config.get('IMAGE', 'WIDTH'))
ENCODER = config.get('MODEL', 'ENCODER')
DECODER = config.get('MODEL', 'DECODER')
BATCH_SIZE = int(config.get('DATALOADER', 'BATCH_SIZE'))
SOFTBCE_WEIGHT = float(config.get('LOSS_WEIGHTS', 'SOFTBCE_WEIGHT'))
DICE_WEIGHT = float(config.get('LOSS_WEIGHTS', 'DICE_WEIGHT'))
FOCAL_WEIGHT = float(config.get('LOSS_WEIGHTS', 'FOCAL_WEIGHT'))
LEARNING_RATE = float(config.get('OPTIMIZER', 'LEARNING_RATE'))
PLATEAU_DECAY_FACTOR = float(config.get('SCHEDULER', 'PLATEAU_DECAY_FACTOR'))
PLATEAU_PATIENCE = int(config.get('SCHEDULER', 'PLATEAU_PATIENCE'))
EARLY_STOPPING_PATIENCE = int(config.get('CALLBACKS', 'EARLY_STOPPING_PATIENCE'))
DEVICE = config.get('TRAINING_LOOP', 'DEVICE')
NUM_EPOCHS = int(config.get('TRAINING_LOOP', 'NUM_EPOCHS'))

# Reversible augmentations (useful for test-time augmentation)
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, border_mode=1, rotate_limit=30),
        albu.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, border_mode=1, always_apply=True), # border_mode=cv2.BORDER_REPLICATE, all images are 480x640px in size
    ]
    return albu.Compose(train_transform)

# Data paths
x_train_path = config.get('DATA_PATHS', 'X_TRAIN')
y_train_path = config.get('DATA_PATHS', 'Y_TRAIN')
x_val_path = config.get('DATA_PATHS', 'X_VAL')
y_val_path = config.get('DATA_PATHS', 'Y_VAL')

# Datasets
train_dataset = Dataset(x_train_path, y_train_path, augmentation=get_training_augmentation(), preprocessing=get_preprocessing())
val_dataset = Dataset(x_val_path, y_val_path, preprocessing=get_preprocessing())
print("Datasets generated!")

# Model
if os.path.exists(f"out/{DECODER}_{ENCODER}_model.pth"): # pre-trained model is present
    model = load(f"out/{DECODER}_{ENCODER}_model.pth")
    print("Previous model loaded!")
else:
    decoder_map = {
            "manet": smp.MAnet,
            "linknet": smp.Linknet,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "pan": smp.PAN,
            "deeplabv3": smp.DeepLabV3,
            "unet": smp.Unet
        }
    segmentation_model = decoder_map.get(DECODER.lower(), smp.Unet)
    if segmentation_model == smp.Unet:
        try: # try loading imagenet weights if available
            model = segmentation_model(in_channels=1, encoder_name=ENCODER, encoder_weights="imagenet", classes=1, activation=None, decoder_attention_type = 'scse')
            print("Imagenet weights loaded!")
        except:
            model = segmentation_model(in_channels=1, encoder_name=ENCODER, classes=1, activation=None, decoder_attention_type = 'scse')
    else:
        try: # try loading imagenet weights if available
            model = segmentation_model(in_channels=1, encoder_name=ENCODER, encoder_weights="imagenet", classes=1, activation=None)
            print("Imagenet weights loaded!")
        except:
            model = segmentation_model(in_channels=1, encoder_name=ENCODER, classes=1, activation=None)
    print("Model generated!")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE // 4, shuffle=False)

# Joint loss
softbce_loss = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1) # they all take logits as input
focal_loss = smp.losses.FocalLoss(mode="binary", alpha=0.2, gamma=2.0)
dice_loss = smp.losses.DiceLoss(mode="binary")
joint_loss = JointLoss(losses=[softbce_loss, dice_loss, focal_loss], weights=[SOFTBCE_WEIGHT, DICE_WEIGHT, FOCAL_WEIGHT]) # ref: https://arxiv.org/ftp/arxiv/papers/2209/2209.00729.pdf

# Metrics
activation = "sigmoid"
metrics = [
    smp_utils.metrics.IoU(activation=activation, threshold=0.5), # threshold is for target binarization
    smp_utils.metrics.Precision(activation=activation), # PR are pixel-wise
    smp_utils.metrics.Recall(activation=activation),
    smp_utils.metrics.Fscore(activation=activation),
]

# Optimizer
optimizer = AdamW([dict(params=model.parameters(), lr=LEARNING_RATE)])

# TODO: Load scheduler status if training is resumed
# Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=PLATEAU_DECAY_FACTOR, patience=PLATEAU_PATIENCE, verbose=True)

# TODO: Load callbacks status if training is resumed
# Callbacks
early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)

# Train epoch
train_epoch = smp_utils.train.TrainEpoch(
    model,
    loss=joint_loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# Validation epoch
val_epoch = smp_utils.train.ValidEpoch(
    model,
    loss=joint_loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# (Optional) Resuming previous training
train_logs_path = f"out/{DECODER}_{ENCODER}_train_logs.csv"
val_logs_path = f"out/{DECODER}_{ENCODER}_val_logs.csv"
model_path = f"out/{DECODER}_{ENCODER}_model.pth"

train_logs, val_logs, last_epoch, max_iou_score = load_logs(train_logs_path, val_logs_path)
if train_logs == []:
    print("New training started!")
else:
    print("Previous training resumed!")


# Training loop
for epoch in range(last_epoch, NUM_EPOCHS):
    print(f"\nEpoch: {epoch+1}/{NUM_EPOCHS}")
    # Running epochs
    current_train_logs = train_epoch.run(train_loader)
    current_val_logs = val_epoch.run(val_loader)

    # Model saving
    iou_score = current_val_logs["iou_score"]
    if max_iou_score < iou_score:
        max_iou_score = iou_score
        save(model, model_path)
        print("Model saved!")

    # Log saving
    train_logs.append(current_train_logs)
    val_logs.append(current_val_logs)
    pd.DataFrame(train_logs).to_csv(train_logs_path, index=False)
    pd.DataFrame(val_logs).to_csv(val_logs_path, index=False)

    # Early stopping
    validation_loss = current_val_logs["joint_loss"]
    if early_stopper.early_stop(validation_loss):
        break
    scheduler.step(validation_loss)
