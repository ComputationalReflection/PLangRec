#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that trains a transformer-based neural network for language classification, following the approach in
https://www.tensorflow.org/text/tutorials/transformer
"""

import logging
import os
import time

import keras
import tensorflow as tf
import numpy as np

from configuration import DATA_PCT, VOCABULARY_SIZE, EMBEDDING_DIM, PICKLE_FILE_NAMES_7M, MODELS_DIR, NUMBER_OF_CLASSES, \
    PICKLE_FILE_NAMES_70M
from data import load_data, select_first_in_list
from utils import compile_model, train_model, evaluate_model, train_model_lazy, save_model, f1_m, precision_m, recall_m
from model_tf import EncoderClassifierModel

def main():
    file_names = PICKLE_FILE_NAMES_7M
    (x_train, y_train), (x_val, y_val) = load_data(file_names)
    #DATA_PCT = 0.01
    # x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
    #                                   select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)

    print(f"Shape of x_train: {x_train.shape}.")

    model_file_name = MODELS_DIR + file_names['model']
    if os.path.exists(model_file_name):
        print(f"Loading model from '{model_file_name}'.")
        model = keras.models.load_model(model_file_name, custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})
    else:
        model = EncoderClassifierModel(n_attention_heads=24, n_transformer_blocks=4,
                                       n_neurons_hidden_dense_layer=2048, dropout_factor=0,
                                       vocabulary_size=VOCABULARY_SIZE,
                                       embedding_dim=EMBEDDING_DIM, number_of_classes=NUMBER_OF_CLASSES,
                                       n_neurons_hidden_classifier=1024)
        model.build(x_train.shape)

    model.summary()

    # ------------ Model training ------------------

    #compile_model(model, "adam", "categorical_crossentropy", ["accuracy"])
    #train_model_lazy(model, patience=50, verbose=2, batch_size=1024, epochs=500,
    #                           x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
    #                           early_stop_monitor="val_loss", dir=MODELS_DIR)

    compile_model(model, "adam", "categorical_crossentropy", ["accuracy", "f1"])
    history = train_model_lazy(model, patience=50, verbose=2, batch_size=1024, epochs=500,
                               x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                               early_stop_monitor="val_loss", model_file_name=model_file_name)

    # ------------ Model evaluation ------------------

    metrics = evaluate_model(model, x_val, y_val)


if __name__ == "__main__":
    print("GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()
