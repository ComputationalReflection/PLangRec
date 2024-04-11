#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that loads datasets from files
"""

import pickle
from typing import Dict

def load_data(file_names:Dict[str,str]):
    with open(file_names['x_train'], 'rb') as handle:
        x_train = pickle.load(handle)
    with open(file_names['y_train'], 'rb') as handle:
        y_train = pickle.load(handle)
    with open(file_names['x_val'], 'rb') as handle:
        x_val = pickle.load(handle)
    with open(file_names['y_val'], 'rb') as handle:
        y_val = pickle.load(handle)
    return (x_train, y_train), (x_val, y_val)


def select_first_in_list(iterable, pencentage: float):
    if pencentage >= 1:
        return iterable
    return iterable[:int(len(iterable)*pencentage)]


def show_data(x_train, y_train, x_val, y_val):
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")

