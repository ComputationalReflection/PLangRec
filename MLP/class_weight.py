#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module calculates class weights for the MLP training by first extracting class labels from the training dataset.
Then, it computes class weights using scikit-learn compute_class_weight function to address
class imbalance. Finally, it returns a dictionary mapping class indices to their respective weights."""

from lazy_load import load_ds_lazy
from parameters import *
from pickle_load import pickle_to_tensor
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any


def get_class_weight(ds_train_y) -> Dict[int, Any]:
    class_labels = np.argmax(ds_train_y, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
    cw_dict = {}
    for lang_index in range(0, class_weights.shape[0]):
        cw_dict[lang_index] = class_weights[lang_index]
    return cw_dict


if __name__ == '__main__':
    for config in process_args():
        ARG_MAP = config
        if ARG_MAP[LAZY_LOAD]:
            train_ds, val_ds = load_ds_lazy(ARG_MAP[BATCH_SIZE], ARG_MAP[N_LABELS], ARG_MAP[EPOCHS])
        else:
            train_ds, val_ds = get_pickle_paths()
            train_y = pickle_to_tensor(train_ds + "_labels")
            print(f"{get_class_weight(train_y)} vector for {ARG_MAP[N_SAMPLES]} samples")
