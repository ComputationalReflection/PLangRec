#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This Python script is designed to read CSV files containing rows of source code encoded as integers (representing
the UTF-8 codes of each character). It uses TensorFlow's tf.data.Dataset for lazy processing. Each CSV file is loaded
into a dataset, where each row is split by commas and converted into a dataset of integer vectors. The script also
supports shuffling files and interleaving with shuffling for better randomness. Additionally, it maps each line to a
tuple of input integer vector and its integer label, batches the data and finally prefetches for optimal performance.
"""


import tensorflow as tf
from parameters import *

N_CORES = tf.data.AUTOTUNE


def load_ds_lazy(batch_size: int, n_labels: int, epochs: int) -> (tf.data.Dataset, tf.data.Dataset):
    train_name, valid_name = get_csv_paths()
    return csv_to_ds(train_name, batch_size, n_labels, epochs), csv_to_ds(valid_name, batch_size, n_labels, epochs)


def csv_to_int_ds(csv_path: str, shuffle_files: bool = False, inter_with_shuffle: bool = False) -> tf.data.Dataset:
    CYCLE = 10_000
    MAX_LINES_PER_FILE = 1_000
    ds = tf.data.Dataset.list_files(file_pattern=csv_path, shuffle=shuffle_files)
    if inter_with_shuffle:
        ds = ds.interleave(lambda file_path: tf.data.TextLineDataset(file_path)
                           .shuffle(buffer_size=MAX_LINES_PER_FILE, reshuffle_each_iteration=False),
                           num_parallel_calls=N_CORES, deterministic=False, cycle_length=CYCLE, block_length=1)
    else:
        ds = ds.interleave(lambda file_path: tf.data.TextLineDataset(file_path), deterministic=True,
                           num_parallel_calls=N_CORES)
    CSV_SEPARATOR = ","
    return ds.map(lambda line: tf.strings.to_number(tf.strings.split(line, CSV_SEPARATOR), tf.dtypes.int32),
                  num_parallel_calls=N_CORES)


def csv_to_ds(csv_path: str, batch_size: int, n_labels: int, epochs: int) -> tf.data.Dataset:
    ds = csv_to_int_ds(csv_path)
    ds = ds.map(lambda line: (line[:-n_labels], line[-n_labels:]), num_parallel_calls=N_CORES)
    ds = ds.batch(batch_size, drop_remainder=False, num_parallel_calls=N_CORES, deterministic=False).repeat(
        epochs).prefetch(tf.data.AUTOTUNE)
    print(f"DS {csv_path} Loaded")

    return ds


if __name__ == '__main__':
    for config in process_args():
        ARG_MAP = config
        load_ds_lazy(ARG_MAP[BATCH_SIZE], ARG_MAP[N_LABELS], ARG_MAP[EPOCHS])
