#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple script to perform the evaluation of several neural models under the test set."""

from pickle_load import pickle_to_numpy
import tensorflow as tf
import sys
import numpy
from typing import List

if __name__ == '__main__':
    N_LANGS: int = 21
    ds_path: str = sys.argv[1]
    model_paths: List[str] = sys.argv[2:]
    for model_path in model_paths:
        print(f"Model:\t{model_path} against Dataset:\t{ds_path}")
        model = tf.keras.models.load_model(model_path)
        test_X = pickle_to_numpy(ds_path)
        test_Y = pickle_to_numpy(ds_path + "_labels")
        test_Y = numpy.argmax(test_Y, axis=1)
        pred_Y = model.predict(test_X)
        pred_Y = numpy.argmax(pred_Y, axis=1)
