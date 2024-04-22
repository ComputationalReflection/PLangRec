#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is designed to generate the final dataset of lines of code by reading a stored CSV containing UFT-8
integer codes for each character in the line. The main steps include shuffling the dataset, separating the input from
its labels, padding sequences, truncating long inputs, substituting out-of-vocabulary symbols, transforming integers
into one-hot vectors and stratifying the dataset to ensure balanced class distribution if specified. The generate
function orchestrates the generation pipeline for both the training and validation (or test) datasets. For each
dataset, it performs the actual preprocessing steps such as reading CSVs, stratifying the dataset, transforming it
into a numeric format suitable for training, and finally storing the preprocessed data either as pickles or as
processed CSV file(s).  The run() function manages several executions based on different configurations specified by
command-line arguments and documented in the parameters.py module. """

import tensorflow as tf
import datetime
from parameters import *
from pickle_store import csv_to_pickles, ds_to_pickles
from lazy_load import csv_to_int_ds
from math import ceil
from os import path

N_CORES = tf.data.AUTOTUNE


def generate(train_path: str, valid_path: str, sample_size: int, n_labels: int, max_len: int, vocab_size: int,
             balanced: bool, shuffle: bool = True, one_hot: bool = False, val_size: int = None,
             oov_chars: int = None, lookup_path: str = None, pickle_storage: bool = False) -> None:
    if oov_chars == None:
        lookup_layer = None

    train_name, val_name = get_csv_paths()
    train_pickle, val_pickle = get_pickle_paths()

    generate_single_ds(valid_path, val_name, val_size, n_labels, max_len, vocab_size, balanced,
                       one_hot, oov_chars, lookup_layer, False, val_pickle if pickle_storage else None)
    if not ARG_MAP[GENERATE_ONLY_VALID]:
        generate_single_ds(train_path, train_name, sample_size, n_labels, max_len, vocab_size, balanced,
                           one_hot, oov_chars, lookup_layer, shuffle, train_pickle if pickle_storage else None)


def stratification(ds: tf.data.Dataset, n_samples: int, balanced: bool, shuffle: bool) -> tf.data.Dataset:
    weights = [0.06312472714285715, 0.09830808428571429, 0.044694694285714286, 0.03602075428571429,
               0.04801155142857143, 0.08796147, 0.03146058285714286, 0.03401188857142857, 0.029402557142857142,
               0.04477434142857143, 0.03890795714285714, 0.055149818571428574, 0.03655892571428571,
               0.08798141714285715, 0.038196754285714285, 0.04105453857142857, 0.03996427142857143,
               0.03307634714285714, 0.035901602857142854, 0.03629995285714286, 0.039137762857142856]
    print(f"Received_size{n_samples}")
    n_labels = len(weights)
    if balanced:
        weights = [1.0 / n_labels] * n_labels
        samples_per_class = [ceil(n_samples / n_labels)] * n_labels
        op_name = f"balancing"
    else:
        samples_per_class = [ceil(i * n_samples) for i in weights]
        op_name = "stratification"
    new_sample_size: int = sum(samples_per_class)
    if ARG_MAP[PICKLE_SIZE] == ARG_MAP[N_SAMPLES] and new_sample_size > ARG_MAP[N_SAMPLES]:
        ARG_MAP[PICKLE_SIZE] = new_sample_size

    print(f"samples per class {samples_per_class}, ({weights})")
    print(f"theoretical total after {op_name}:\t{new_sample_size} vs {n_samples} required by user")

    if shuffle:
        datasets = [ds.filter(lambda _, label: label == i).take(samples_per_class[i]) for i in range(n_labels)]
        if get_device_name() == "GPU":
            return tf.data.experimental.sample_from_datasets(datasets, weights)
        else:
            return tf.data.Dataset.sample_from_datasets(datasets, weights, stop_on_empty_dataset=True)

    stratified_ds = ds.filter(lambda _, label: label == 0).take(samples_per_class[0])
    for i in range(1, n_labels):
        stratified_ds = stratified_ds.concatenate(ds.filter(lambda _, label: label == i).take(samples_per_class[i]))
    return stratified_ds


def generate_single_ds(ds_path: str, name: str, sample_size: int, n_labels: int, max_len: int, vocab_size: int,
                       balanced: bool, one_hot: bool = False, oov_chars: int = None, lookup_layer=None
                       , shuffle: bool = False, pickle_storage: str = None) -> None:
    print(f"processing {name}...{datetime.datetime.now().time()}")

    print(f"Reading CSVs...{datetime.datetime.now().time()}")
    ds = csv_to_int_ds(ds_path, False, False)
    ds = ds.filter(lambda line: tf.shape(line)[0] >= 11)
    ds = ds.map(lambda line: (line[:-1], line[-1]))

    print(f"Starting stratification{datetime.datetime.now().time()}")
    ds = stratification(ds, sample_size, balanced, shuffle)

    print(f"Starting transformation{datetime.datetime.now().time()}")
    ds = transform_to_numeric_ds(ds, n_labels, max_len, vocab_size, one_hot, oov_chars, lookup_layer)
    print(f"Transformation finished{datetime.datetime.now().time()}")

    if pickle_storage is not None:
        print(f"storing {name} directly to pickle...{datetime.datetime.now().time()}")

        ds_labels = ds.map(lambda line, target: target, num_parallel_calls=N_CORES)
        ds = ds.map(lambda line, target: line, num_parallel_calls=N_CORES)
        ds_to_pickles(ds, ds_labels, ARG_MAP[PICKLE_SIZE], pickle_storage)
    else:
        print(f"storing {name} into single processed csv...{datetime.datetime.now().time()}")
        ds = from_tensor_to_string(ds)
        write_in_one_csv(ds, name, sample_size, batch_size=ARG_MAP[BATCH_SIZE], shuffle=False)

    print(f"Storage finished...{datetime.datetime.now().time()}")


def from_tensor_to_string(ds) -> tf.data.Dataset:
    CSV_SEPARATOR = ","
    return ds.map(lambda line, label: tf.strings.reduce_join(tf.strings.as_string(line),
                                                             separator=CSV_SEPARATOR) + CSV_SEPARATOR + tf.strings.reduce_join(
        tf.strings.as_string(label), separator=CSV_SEPARATOR))


def transform_to_numeric_ds(ds: tf.data.Dataset, n_labels: int, max_len: int, vocab_size: int, one_hot: bool = False,
                            oov_chars: int = None, lookup_layer=None) -> tf.data.Dataset:
    print("one hot LABEL:\t{}".format(not ARG_MAP[INT_LABEL]))
    if not ARG_MAP[INT_LABEL]:
        ds = ds.map(lambda line, target: (line, tf.one_hot(indices=tf.cast(target, tf.dtypes.uint8), depth=n_labels,
                                                           dtype=tf.dtypes.int8)), num_parallel_calls=N_CORES)
    ds = ds.map(lambda line, target: (line[:max_len], target), num_parallel_calls=N_CORES)

    if oov_chars is None:
        ds = process_char_numbers(ds)

    ds = ds.map(lambda line, label: (tf.concat([line, tf.zeros([max_len - tf.shape(line)[0]],
                                                               dtype=tf.dtypes.int32)], axis=0), label),
                num_parallel_calls=N_CORES)
    if one_hot:
        ds = ds.map(lambda line, label: (tf.one_hot(indices=line, depth=vocab_size, dtype=tf.dtypes.int32), label),
                    num_parallel_calls=N_CORES)

    return ds


def process_char_numbers(ds: tf.data.Dataset) -> tf.data.Dataset:
    EMPTY_SPACES = MIN_POS - 2
    ds = ds.map(lambda line, label: (line - EMPTY_SPACES, label), num_parallel_calls=N_CORES)
    OOV_CHAR = 1
    ds = ds.map(lambda line, label: (tf.where(line < MIN_POS - EMPTY_SPACES, OOV_CHAR, line), label),
                num_parallel_calls=N_CORES)
    ds = ds.map(lambda line, label: (tf.where(line >= MAX_POS - EMPTY_SPACES, OOV_CHAR, line), label),
                num_parallel_calls=N_CORES)
    return ds


def write_in_one_csv(ds: tf.data.Dataset, csv_name: str, sample_size: int, batch_size: int = 16_384, shuffle: bool = False):
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    with open(csv_name, 'w') as csvfile:
        i: int = 1
        total: int = 0
        for batch in ds:
            if shuffle:
                batch = tf.random.shuffle(batch)
            file_content = ""
            if i > 1:
                file_content += "\n"
            file_content += str(tf.strings.reduce_join(batch, separator="\n").numpy(), encoding="ascii")
            csvfile.write(file_content)
            i += 1
            total += batch.shape[0]
            print(
                f"{batch.shape[0]} instances written. {total} samples written {int(total * 100 / sample_size)}%")


def run() -> None:
    print(f"starting whole process at...{datetime.datetime.now().time()}")
    n_labels = ARG_MAP[N_LABELS]
    if not ARG_MAP[PICKLE_ONLY]:
        generate(ARG_MAP[RAW_CSV_TRAIN_PATH], ARG_MAP[RAW_CSV_VALID_PATH], ARG_MAP[N_SAMPLES], n_labels,
                 ARG_MAP[MAX_LEN], ARG_MAP[VOCAB_SIZE], balanced=ARG_MAP[BALANCED], shuffle=ARG_MAP[SHUFFLE],
                 one_hot=ARG_MAP[ONE_HOT], val_size=ARG_MAP[N_VALID], oov_chars=ARG_MAP[LOOKUP_OOV],
                 lookup_path=get_lookup_file())
    print("Storing pickle....")
    if not ARG_MAP[LAZY_LOAD]:
        train_name, val_name = get_csv_paths()
        train_pickle, val_pickle = get_pickle_paths()
        if path.exists(val_name) and path.exists(train_name):
            csv_to_pickles(val_name, ARG_MAP[PICKLE_SIZE], val_pickle, n_labels)
            if not ARG_MAP[GENERATE_ONLY_VALID]:
                csv_to_pickles(train_name, ARG_MAP[PICKLE_SIZE], train_pickle, n_labels)
        else:
            print(f"File {val_pickle} not found, and pickle_only set to True,"
                  f" so preprocessing and storing directly as pickle")
            generate(ARG_MAP[RAW_CSV_TRAIN_PATH], ARG_MAP[RAW_CSV_VALID_PATH], ARG_MAP[N_SAMPLES], n_labels,
                     ARG_MAP[MAX_LEN], ARG_MAP[VOCAB_SIZE], balanced=ARG_MAP[BALANCED], shuffle=ARG_MAP[SHUFFLE],
                     one_hot=ARG_MAP[ONE_HOT], val_size=ARG_MAP[N_VALID], oov_chars=ARG_MAP[LOOKUP_OOV],
                     lookup_path=get_lookup_file(), pickle_storage=True)
    print(f"whole process finished at...{datetime.datetime.now().time()}")


if __name__ == '__main__':
    for config in process_args():
        set_arg_map(config)
        run()
