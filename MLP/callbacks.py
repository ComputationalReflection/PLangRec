#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module defines a custom callback PrintAndSaveStats to monitor and save various statistics during the training of a
MLP model. The callback logs information such as epoch timings, accuracy, loss, and metrics like precision and recall.
It also computes aggregates like total training time and best accuracy achieved. Additionally, it writes these
statistics to a file and logs them for TensorBoard visualization. The get_callbacks function generates a list of
callbacks including Early Stopping, model checkpointing, the custom PrintAndSaveStats, and TensorBoard logging,
tailored for a specific model with given parameters.
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
import datetime
import time
from parameters import get_tensorboard_path


class PrintAndSaveStats(tf.keras.callbacks.Callback):

    def __init__(self, model_name):
        self.epoch_time_start = None
        self.model_name = model_name
        self.total_time = 0
        self.last_epoch = 1
        self.best_acc = 0
        self.best_epoch = 1
        self.first_acc = 0
        self.last_acc = 0
        self.last_loss = 0
        self.last_f1_micro = 0
        self.last_f1_macro = 0
        self.last_precision = 0
        self.last_recall = 0

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs):
        epoch += 1
        if epoch == 1:
            self.first_acc = logs["val_accuracy"]
        print('Epoch {} finished at {}'.format(epoch, datetime.datetime.now().time()))
        print(f"Printing log object:\n{logs}")
        elapsed_time = int((time.time() - self.epoch_time_start))
        print(f"Elaspsed time: {elapsed_time}")
        if logs["loss"] != 0:
            print("val/train loss: {:.2f}".format(logs["val_loss"] / logs["loss"]))
        if logs["accuracy"] != 0:
            print("val/train acc: {:.2f}".format(logs["val_accuracy"] / logs["accuracy"]))
        file1 = open(get_history_path(self.model_name), "a")  # append mode
        SEPARATOR = ";"
        file1.write(str(epoch) + SEPARATOR + str(datetime.datetime.now().time()) + SEPARATOR +
                    str(elapsed_time) + SEPARATOR + str(logs["accuracy"]) + SEPARATOR +
                    str(logs["val_accuracy"]) + SEPARATOR + str(logs["loss"]) + SEPARATOR + str(logs["val_loss"])
                    + "\n")
        file1.close()
        self.compute_aggregates(elapsed_time, logs["val_accuracy"], epoch)

        self.last_acc = logs["val_accuracy"]
        self.last_loss = logs["val_loss"]
        # self.last_f1_micro = logs["val_f1_micro"]
        # self.last_f1_macro = logs["val_f1_macro"]
        self.last_precision = logs["val_precision"]
        self.last_recall = logs["val_recall"]
        with tf.summary.create_file_writer(get_tensorboard_path()).as_default():
            tf.summary.scalar("val_accuracy", logs["val_accuracy"], step=epoch)
            tf.summary.scalar("val_loss", logs["val_loss"], step=epoch)
            tf.summary.scalar("train_accuracy", logs["accuracy"], step=epoch)
            tf.summary.scalar("train_loss", logs["loss"], step=epoch)
            tf.summary.scalar("time", elapsed_time, step=epoch)
            tf.summary.scalar("precision", logs["val_precision"], step=epoch)
            tf.summary.scalar("recall", logs["val_recall"], step=epoch)
            # tf.summary.scalar("f1_macro",  logs["val_f1_macro"], step=epoch)
            # tf.summary.scalar("f1_micro",  logs["val_f1_micro"], step=epoch)

    def compute_aggregates(self, elapsed_time: int, val_acc, epoch: int):
        self.total_time += elapsed_time
        self.last_epoch = epoch
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = epoch

    def get_stats(self):
        return [int(self.total_time / self.last_epoch), self.first_acc, self.best_acc, self.best_epoch, self.last_epoch]


def get_history_path(model_name: str):
    return model_name + "_history.csv"


def get_best_model_path(model_name: str):
    return model_name + "_checkpoint.h5"


def get_callbacks(model_name: str, early_patience: int) -> list:
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=early_patience,
                                   restore_best_weights=True, verbose=1)
    save_best_model = ModelCheckpoint(get_best_model_path(model_name), save_best_only=True, monitor="val_loss", verbose=1)
    save_model_stats = PrintAndSaveStats(model_name)
    tensorboard = TensorBoard(get_tensorboard_path())
    return [save_best_model, save_model_stats, early_stopping, tensorboard]
