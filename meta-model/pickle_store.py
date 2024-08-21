#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module devoted to store datasets as serialized numpy arrays in pickle files. The supported inputs are CSV files or
tf.data.Dataset."""

import pickle
import tensorflow as tf


N_CORES = tf.data.AUTOTUNE


def ds_to_pickles(ds: tf.data.Dataset, ds_labels: tf.data.Dataset, instances_per_pickle: int,
                  pickle_name: str):
    print(f"Starting pickle storage at {pickle_name}")
    ds = ds.batch(instances_per_pickle).prefetch(tf.data.AUTOTUNE)
    ds_labels = ds_labels.batch(instances_per_pickle).prefetch(tf.data.AUTOTUNE)
    print(f"Batching finished")
    single_ds_to_pickles(ds_labels, pickle_name + "_labels")
    print(f"labels stored")
    single_ds_to_pickles(ds, pickle_name)
    print(f"attributes stored")


def single_ds_to_pickles(ds, pickle_name: str):
    index = 0
    for batch in ds:
        numpy_to_pickle(batch.numpy(), pickle_name + "_" + str(index))
        index += 1


def numpy_to_pickle(numpy, pickle_name: str):
    with open(pickle_name, 'wb') as pickle_file:
        pickle.dump(numpy, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def lookup_pickle(csv_path: str, n_samples: int, max_len: int, pickle_name: str):
    ds = csv_to_int_ds(csv_path).take(n_samples)
    ds = ds.map(lambda line: line[:-1])
    ds = ds.map(lambda line: line[:max_len], num_parallel_calls=N_CORES)

    ds = ds.map(lambda line: tf.concat([line, tf.zeros([max_len - tf.shape(line)[0]],
                                                       dtype=tf.dtypes.int32) - 2], axis=0),
                num_parallel_calls=N_CORES)
    ds = ds.batch(n_samples).prefetch(tf.data.AUTOTUNE)
    single_ds_to_pickles(ds, pickle_name)
