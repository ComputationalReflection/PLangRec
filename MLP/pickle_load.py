#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module devoted to load serialized datasets in pickle format into memory."""

import pickle
import tensorflow as tf


def load_ds_from_pickle(ds_path: str, pickle_index: int = 0) -> (tf.Tensor, tf.Tensor):
    return pickle_to_tensor(ds_path, pickle_index), pickle_to_tensor(ds_path + "_labels", pickle_index)


def pickle_to_tensor(file_name: str, pickle_index: int = 0):
    numpy = pickle_to_numpy(file_name + ("" if pickle_index == 0 else "_" + str(pickle_index)))
    with tf.device('/CPU:0'):
        tensor = tf.constant(numpy)
        return tensor


def pickle_to_numpy(file_name: str):
    with open(file_name, 'rb') as pickle_file:
        numpy = pickle.load(pickle_file)
    print(f"loaded {numpy.shape} numpy pickle.")
    return numpy
