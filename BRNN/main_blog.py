#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that trains a transformer-based neural network for language classification, following the approach in
https://blog.paperspace.com/transformers-text-classification/
"""


import pickle
from typing import Dict

import keras
import numpy as np
import math
import tensorflow as tf
import os

from configuration import PICKLE_FILE_NAMES_70M, DATA_PCT, N_ATTENTION_HEADS, N_NEURONS_HIDDEN_DENSE_LAYER_ENCODER, \
    MAX_CHARS_PER_LINE, VOCABULARY_SIZE, NUMBER_OF_CLASSES, N_NEURONS_HIDDEN_DENSE_LAYER_CLASSIFIER, EMBEDDING_DIM, \
    PICKLE_FILE_NAMES_700M, MODELS_DIR, PICKLE_FILE_NAMES_7M, LOG_DIR
from data import load_data, select_first_in_list
from model_blog import create_model, create_deep_model, create_deep_model
from utils import compile_model, train_model, evaluate_model, train_model_lazy, \
    save_model, f1_m, precision_m, recall_m, disable_keras_warning_messages, create_log_file


def main_test():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_70M)
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                     select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)
    model = create_model(n_attention_heads=N_ATTENTION_HEADS,
                         n_neurons_hidden_dense_layer=N_NEURONS_HIDDEN_DENSE_LAYER_ENCODER,
                         max_char_per_line=MAX_CHARS_PER_LINE, vocabulary_size=VOCABULARY_SIZE,
                         embedding_dim=EMBEDDING_DIM, number_of_classes=NUMBER_OF_CLASSES, dropout_factor=0,
                         n_neurons_hidden_classifier=N_NEURONS_HIDDEN_DENSE_LAYER_CLASSIFIER)
    model.summary()
    compile_model(model, "adam", "categorical_crossentropy", ["accuracy"])
    history = train_model(model, 'val_loss', patience=0, verbose=1, batch_size=64, epochs=1,
                          x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    # ------------ Model evaluation ------------------
    evaluate_model(model, x_val, y_val)


def main_simple():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_7M)
    DATA_PCT = 0.001
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)
    model = create_model(n_attention_heads=N_ATTENTION_HEADS,
                         n_neurons_hidden_dense_layer=N_NEURONS_HIDDEN_DENSE_LAYER_ENCODER,
                         max_char_per_line=MAX_CHARS_PER_LINE, vocabulary_size=VOCABULARY_SIZE,
                         embedding_dim=EMBEDDING_DIM, number_of_classes=NUMBER_OF_CLASSES, dropout_factor=0,
                         n_neurons_hidden_classifier=N_NEURONS_HIDDEN_DENSE_LAYER_CLASSIFIER)
    model.summary()
    compile_model(model, "adam", "categorical_crossentropy", ["accuracy"])
    history = train_model(model, 'val_loss', patience=50, verbose=1, batch_size=64, epochs=100,
                          x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, file_name="output.txt")
    # ------------ Model evaluation ------------------
    evaluate_model(model, x_val, y_val)


def main_deep():
    BATCH_SIZE, N_ATTENTION_HEADS, N_TRANS_BLOCKS, N_NEURONS_HIDDEN_DENSE_LAYER_ENCODER, DROP_OUT,\
        N_NEURONS_HIDDEN_DENSE_LAYER_CLASSIFIER, N_CLASS_LAYERS, LEARNING_RATE = 64, 24, 4, 1024, 0, 512, 2, 0.001
    file_names = PICKLE_FILE_NAMES_700M
    (x_train, y_train), (x_val, y_val) = load_data(file_names)

    DATA_PCT = 1

    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)
    #gc.collect()

    print(f"Len of x_train: {x_train.shape[0]:,}. Shape of x_train: {x_train.shape}.")
    model_file_name = MODELS_DIR + file_names['model']
    # if os.path.exists(model_file_name):
    #     print(f"Loading model from '{model_file_name}'.")
    #     model = keras.models.load_model(model_file_name, custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})
    # else:

    BATCH_SIZE = 512
    N_ATTENTION_HEADS, N_TRANS_BLOCKS, N_NEURONS_HIDDEN_DENSE_LAYER_ENCODER,  \
        N_NEURONS_HIDDEN_DENSE_LAYER_CLASSIFIER, N_CLASS_LAYERS = 48, 4, 512, 1024, 2
    DROP_OUT = 0.2

    model = create_deep_model(n_attention_heads=N_ATTENTION_HEADS, n_transformer_blocks=N_TRANS_BLOCKS,
, ,                  n_neurons_hidden_dense_layer=N_NEURONS_HIDDEN_DENSE_LAYER_ENCODER, dropout_factor=DROP_OUT,
, ,                  max_words_per_review=MAX_CHARS_PER_LINE, vocabulary_size=VOCABULARY_SIZE,
, ,                  embedding_dim=EMBEDDING_DIM, number_of_classes=NUMBER_OF_CLASSES,
, ,                  n_neurons_hidden_classifier=N_NEURONS_HIDDEN_DENSE_LAYER_CLASSIFIER,
, ,                       n_layers_hidden_classifier=N_CLASS_LAYERS)
    model.summary()
    compile_model(model, "adam", "categorical_crossentropy", ["accuracy", "f1"], learning_rate=LEARNING_RATE)
    log_file = create_log_file(LOG_DIR, x_train.shape[0], BATCH_SIZE, N_ATTENTION_HEADS, N_TRANS_BLOCKS, N_NEURONS_HIDDEN_DENSE_LAYER_ENCODER, DROP_OUT,\
,     N_NEURONS_HIDDEN_DENSE_LAYER_CLASSIFIER, N_CLASS_LAYERS, LEARNING_RATE )
    history = train_model_lazy(model, patience=50, verbose=2, batch_size=BATCH_SIZE, epochs=500,
                               x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                               early_stop_monitor="val_loss", model_file_name=model_file_name,
                               log_file_name=log_file, patience_lr=0)
    # ------------ Model evaluation ------------------
    print("Evaluation of val set:")
    metrics = evaluate_model(model, x_val, y_val)


if __name__ == "__main__":
    print("GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    disable_keras_warning_messages()
    main_deep()  # 92.5%
    #main_simple()
    
