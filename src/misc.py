import csv
import os


def convert_values_to_float(row):
    float_row = {}
    for key, value in row.items():
        try:
            float_value = float(value)
        except ValueError:
            float_value = value
        float_row[key] = float_value
    return float_row

def load_logs(train_logs_path, val_logs_path):
    train_logs = []
    val_logs = []
    last_epoch = 0
    max_iou_score = 0.0

    if os.path.exists(train_logs_path) and os.path.exists(val_logs_path):
        with open(train_logs_path, 'r') as file:
            train_logs = [convert_values_to_float(row) for row in csv.DictReader(file)]
        with open(val_logs_path, 'r') as file:
            val_logs = [convert_values_to_float(row) for row in csv.DictReader(file)]
            last_epoch = len(val_logs)
            max_iou_score = max(val_log["iou_score"] for val_log in val_logs)
    
    return train_logs, val_logs, last_epoch, max_iou_score
