#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is dedicated to training a Multilayer Perceptron (MLP) TensorFlow model. It first loads or compiles
the model by calling the model module, based on provided parameters such as  number of dense layers, embedding size,
number of neurons per layer, activation function, dropout rate, initializers, and learning rate. Then, it trains the
model using the specified training and validation datasets, along with parameters like the initial epoch (used only
to resume a previous training process), number of epochs, and patience for Early Stopping. It orchestrates the whole
training process by handling exceptions, logging metrics, and storing the final model. The run() function manages
several executions based on different configurations specified by command-line arguments and documented in the
parameters.py module."""

from tensorflow.keras.models import Sequential
import tensorflow as tf
from parameters import *
from model import compile_model
from store_model import store_last_model, get_last_model_path
from lazy_load import load_ds_lazy
from callbacks import get_callbacks
from store_stats import append_stats_after_train
from pickle_load import load_ds_from_pickle
from class_weight import get_class_weight
from tensorboard.plugins.hparams import api as hp
import traceback


def fit(ds_train_x, ds_train_y, ds_val, model: Sequential, epochs: int, callbacks: list, batch_size: int = None
        , steps_per_epoch: int = None, val_steps: int = None, class_weight=None, initial_epoch: int = 0):
    return model.fit(ds_train_x, ds_train_y, epochs=epochs, verbose=2, validation_data=ds_val,
                     use_multiprocessing=True, callbacks=callbacks, steps_per_epoch=steps_per_epoch,
                     batch_size=batch_size, validation_steps=val_steps, class_weight=class_weight,
                     initial_epoch=initial_epoch)


def single_pickle_train(ds_train, ds_val, model: Sequential, epochs: int, callbacks: list, batch_size: int
                        , class_weight: tf.Tensor, initial_epoch: int):
    return fit(ds_train_x=ds_train[0], ds_train_y=ds_train[1], ds_val=ds_val, model=model, epochs=epochs,
               callbacks=callbacks, batch_size=batch_size, class_weight=class_weight, initial_epoch=initial_epoch)


def several_pickle_train(ds_train: str, ds_val: str, model: Sequential, epochs: int, callbacks: list, batch_size: int,
                         pickle_size: int, train_size: int):
    ds_val = load_ds_from_pickle(ds_val)
    n_pickles = int(train_size / pickle_size)
    if train_size % pickle_size != 0:
        n_pickles += 1

    for epoch in range(0, epochs):
        for pickle_index in range(0, n_pickles):
            print(f"Training pickle {pickle_index} of epoch {epoch}")
            history = single_pickle_train(load_ds_from_pickle(ds_train, pickle_index), ds_val, model, epochs=1,
                                          callbacks=callbacks, batch_size=batch_size)

    return history


def pickle_train(ds_train: str, ds_val: str, model: Sequential, epochs: int, callbacks: list, batch_size: int,
                 pickle_size: int, train_size: int, balanced: bool, initial_epoch: int):
    if pickle_size >= train_size:
        ds_train = load_ds_from_pickle(ds_train)
        class_weight = None
        if not balanced:
            class_weight = get_class_weight(ds_train[1])
            print(class_weight)

        return single_pickle_train(ds_train, load_ds_from_pickle(ds_val), model, epochs, callbacks,
                                   batch_size, class_weight, initial_epoch)
    else:
        print("YOUR PICKLE SIZE IS LOWER THAN THE NUMBER OF SAMPLES")
        print("balanced not implemented for several pickles...")
        return several_pickle_train(ds_train, ds_val, model, epochs, callbacks, batch_size, pickle_size, train_size)


def lazy_train(ds_train: tf.data.Dataset, ds_val: tf.data.Dataset, train_batches: int, val_batches: int,
               model: Sequential, epochs: int, callbacks: list):
    return fit(ds_train_x=ds_train, ds_train_y=None, ds_val=ds_val, model=model, epochs=epochs, callbacks=callbacks,
               steps_per_epoch=train_batches, val_steps=val_batches)


def train(lazy_load: bool, ds_train, ds_val, train_size: int, valid_size: int, batch_size: int,
          model: Sequential, epochs: int, early_patience: int, pickle_size: int, balanced: bool,
          initial_epoch: int):
    model_name = get_model_name()
    callbacks = get_callbacks(model_name, early_patience)
    history = None
    try:
        train_batches = int(train_size / batch_size) + (0 if train_size % batch_size == 0 else 1)
        print(f"Training with {train_batches} batches of {batch_size} (approx. {train_batches * batch_size} instances)")
        val_batches = int(valid_size / batch_size) + (0 if valid_size % batch_size == 0 else 1)
        print(f"Validation with {val_batches} batches of {batch_size} (approx. {val_batches * batch_size} instances)")

        if lazy_load:
            history = lazy_train(ds_train, ds_val, train_batches, val_batches, model, epochs, callbacks, initial_epoch)
        else:
            history = pickle_train(ds_train, ds_val, model, epochs, callbacks, batch_size, pickle_size, train_size
                                   , balanced, initial_epoch)
    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()

    finally:
        SAVE_STATS_INDEX = 1
        SEPARATOR = "\t"
        print("Doing hp...")

        with tf.summary.create_file_writer(get_tensorboard_path()).as_default():
            hp.hparams(get_hp_param_map())
            tf.summary.scalar("METRIC_ACCURACY", callbacks[SAVE_STATS_INDEX].last_acc, step=1)
            tf.summary.scalar("METRIC_LOSS", callbacks[SAVE_STATS_INDEX].last_loss, step=1)
            tf.summary.scalar("METRIC_PRECISION", callbacks[SAVE_STATS_INDEX].last_precision, step=1)
            tf.summary.scalar("RECALL", callbacks[SAVE_STATS_INDEX].last_recall, step=1)

        print("Storing final model")
        store_last_model(model_name, model, history)
        print("Appending stats...")
        append_stats_after_train(callbacks[SAVE_STATS_INDEX], GLOBAL_CSV, SEPARATOR, model.count_params())


def run() -> None:
    if ARG_MAP[LAZY_LOAD]:
        train_ds, val_ds = load_ds_lazy(ARG_MAP[BATCH_SIZE], ARG_MAP[N_LABELS], ARG_MAP[EPOCHS])
    else:
        train_ds, val_ds = ARG_MAP[DS_TRAIN_PATH], ARG_MAP[DS_VALID_PATH]
    if ARG_MAP[START] == 0:
        main_model = compile_model(ARG_MAP[MAX_LEN], ARG_MAP[VOCAB_SIZE], ARG_MAP[N_LABELS],
                                   ARG_MAP[N_DENSES], ARG_MAP[DENSES_WIDTH], ARG_MAP[EMBEDDING_SIZE],
                                   ARG_MAP[FLATTEN_DOWN], ARG_MAP[DROPOUT], ARG_MAP[ACTIVATION_FUNC],
                                   ARG_MAP[WEIGHT_INITIALIZER], ARG_MAP[L_RATE])
    else:
        main_model = tf.keras.models.load_model(get_last_model_path(get_model_name()))

    train(ARG_MAP[LAZY_LOAD], train_ds, val_ds, ARG_MAP[N_SAMPLES], ARG_MAP[N_VALID], ARG_MAP[BATCH_SIZE], main_model,
          ARG_MAP[EPOCHS], ARG_MAP[PATIENCE], ARG_MAP[PICKLE_SIZE], ARG_MAP[BALANCED], ARG_MAP[START])


if __name__ == '__main__':
    for config in process_args():
        set_arg_map(config)
        run()
