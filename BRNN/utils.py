#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Different utility functions used by other modules
"""

import gc
import os
import pickle
from typing import List

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential, Model
import keras
from datetime import datetime
import os

import numpy as np
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def compile_model(model, optimizer:str, loss:str, metrics:List[str], learning_rate: float=0.001):
    pairs = {'f1': f1_m, 'precision': precision_m, 'recall': recall_m}
    metrics = [pairs[metric] if metric in pairs else metric for metric in metrics]
    #learning_rate =0.001*5.65  # 0.001 = default; 5.65 = sqrt(1024(batchsize)/32(default_batchsize))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999) if optimizer == 'adam' else optimizer
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train_model(model, early_stop_monitor:str, patience: int, verbose:int, batch_size:int, epochs:int,
                x_train, y_train, x_val, y_val, file_name: str, patience_lr: int = 4):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=patience, verbose=verbose,
                                                           restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr, min_lr=0.00001, verbose=verbose)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_name,
        save_weights_only=False, monitor='val_loss', save_best_only=True, verbose=1)
    history = model.fit(x_train, y_train, batch_size=batch_size,
                              epochs=epochs, validation_data=(x_val, y_val),
                    callbacks=[early_stop_callback, model_checkpoint_callback])
    return history


def __generate_data(x, y, batch_size, epochs:int):
    for epoch in range(epochs):
        to_index = 0
        for n in range(x.shape[0]//batch_size):
            from_index, to_index = n*batch_size, (n+1)*batch_size
            yield np.array(x[from_index:to_index]), np.array(y[from_index:to_index])
        yield np.array(x[to_index:]), np.array(y[to_index:])


def train_model_lazy(model, verbose:int, batch_size:int, epochs:int,
                     x_train, y_train, x_val, y_val, csv_file_name: str,
                     model_file_name: str = None, log_file_name: str = None,
                     patience_lr: int = None, early_stop_monitor: str = None, patience: int = None):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=patience, verbose=verbose,
                                                           restore_best_weights=True) if early_stop_monitor else None
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr, min_lr=0.00001, verbose=verbose)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_file_name)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file_name,
                                                                   save_weights_only=False, monitor='val_loss', save_best_only=True, verbose=1)
    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_file_name, separator=",", append=False)
    callbacks = list()
    if csv_file_name: callbacks.append(csv_logger_callback)
    if log_file_name: callbacks.append(tensorboard_callback)
    if model_file_name: callbacks.append(model_checkpoint_callback)
    if early_stop_callback: callbacks.append(early_stop_callback)
    if patience_lr: callbacks.append(reduce_lr_callback)
    model.fit(__generate_data(x_train, y_train, batch_size, epochs=epochs),
                    steps_per_epoch = x_train.shape[0]//batch_size+1,
                              epochs=epochs, validation_data=(x_val, y_val),
                    callbacks=callbacks)

def evaluate_model(model, x_val, y_val):
    results = model.evaluate(x_val, y_val, verbose=1)
    for name, value in zip(model.metrics_names, results):
        print("%s: %f" % (name, value), end=", ")
    print()
    return results

def save_model(dir, model, accuracy):
    file_name = f"{dir}accuracy_{accuracy}_{str(datetime.now())}.pkl"
    file_name = file_name.replace(" ", "_").replace(":", "_")
    with open(file_name, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved with name: {file_name}.")


def disable_keras_warning_messages():
    from sys import platform
    if platform == "linux" or platform == "linux2":
        command = "export TF_CPP_MIN_LOG_LEVEL=3"
        os.system(command)
    elif platform == "darwin":
        # OS X
        command = "export TF_CPP_MIN_LOG_LEVEL=3"
        os.system(command)
    elif platform == "win32":
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable keras warnings


def create_log_file(log_dir:str, individuals: int, batch_size: int, n_attention_heads: int, n_tranformer_blocks: int, n_neurons_encoder: int,
                    drop_out: float, n_neurons_classifier: int, n_class_layers: int, learning_rate: float):
    return f"{log_dir}{individuals}-" \
        f"batch_size_{batch_size}-n_attention_heads_{n_attention_heads}-n_tranformer_blocks_{n_tranformer_blocks}-" \
           f"n_neurons_encoder_{n_neurons_encoder}-drop_out_{drop_out}-n_neurons_classifier_{n_neurons_classifier}-" \
           f"n_class_layers_{n_class_layers}-learning_rate_{learning_rate}"