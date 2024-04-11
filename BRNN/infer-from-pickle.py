#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example use of model inference, by restoring it from a serialized model (using pickle)
"""

import pickle
from utils import evaluate_model
import os

MODELS_DIR = './models/'
PICKLE_FILE_NAMES_7M = {
    "x_train": "pkl/7_000_000_train.pkl",
    "y_train": "pkl/7_000_000_train_labels.pkl",
    "x_val": "pkl/7_000_000_valid.pkl",
    "y_val": "pkl/7_000_000_valid_labels.pkl",
}

def get_file_with_highest_accuracy(dir: str) -> str:
    file_names = os.listdir(dir)
    file_names = list(filter(lambda file_name: file_name.startswith('accuracy'), file_names))
    file_names.sort()
    if len(file_names):
        return dir + file_names[-1]
    return None

def main():
    file_name = get_file_with_highest_accuracy(MODELS_DIR)
    if not file_name:
        return
    print("Loading X for validation...")
    with open(PICKLE_FILE_NAMES_7M['x_val'], 'rb') as handle:
        x_val = pickle.load(handle)
    print("Loading Y validation...")
    with open(PICKLE_FILE_NAMES_7M['y_val'], 'rb') as handle:
        y_val = pickle.load(handle)
    print(f"Loading the model from {file_name} ...")
    with open(file_name, 'rb') as handle:
        model = pickle.load(handle)
    print("Evaluating the model...")
    evaluate_model(model, x_val, y_val)


if __name__ == "__main__":
    main()
