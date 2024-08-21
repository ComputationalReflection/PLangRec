#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module loads code snippet files into TensorFlow datasets and processes them into numeric vectors
for language identification tasks. It preprocesses snippets by splitting them into lines, converting characters
to integers, handling out-of-vocabulary characters, and padding sequences to a fixed length.
The `snippets_to_int_ds` function decodes and splits text files into lines and encodes each line
as a sequence of integers. The `preprocess_int_snippet` function adjusts the integer sequences
and pads them to ensure uniform length. The `process_snippet_ds` function orchestrates these
preprocessing steps. The `get_snippet_ds` function loads snippets from a specified directory
and applies the necessary preprocessing to create a dataset suitable for model training.
"""

import tensorflow as tf

SNIPPET_SOURCE_FOLDER = "C:\\Users\\Oskar\\Desktop\\cosas_pato\\processed_test_snippet_{}"

N_CORES = tf.data.AUTOTUNE


def preprocess_int_snippet(ds: tf.data.Dataset, max_len: int):
    MIN_POS = 32
    MAX_POS = 127
    EMPTY_SPACES = MIN_POS - 2
    ds = ds.map(lambda line, label: (line - EMPTY_SPACES, label), num_parallel_calls=N_CORES)
    OOV_CHAR = 1
    ds = ds.map(lambda line, label: (tf.where(line < MIN_POS - EMPTY_SPACES, OOV_CHAR, line), label),
                num_parallel_calls=N_CORES)
    ds = ds.map(lambda line, label: (tf.where(line >= MAX_POS - EMPTY_SPACES, OOV_CHAR, line), label),
                num_parallel_calls=N_CORES)

    ds = ds.map(lambda lines, label: (tf.map_fn(fn=lambda line: tf.concat([line, tf.zeros([max_len - tf.shape(line)[0]],
                                                                                          dtype=tf.dtypes.int32)],
                                                                          axis=0), elems=lines),
                                      label), num_parallel_calls=N_CORES)
    return ds


def snippets_to_int_ds(ds: tf.data.Dataset, max_len: int):
    ds = ds.map(lambda file, label: (file, tf.cast(label, tf.dtypes.int32)), num_parallel_calls=N_CORES)
    ds = ds.map(lambda file, label: (tf.strings.unicode_decode(file, "UTF-16LE"), label), num_parallel_calls=N_CORES)
    ds = ds.map(lambda file, label: (tf.strings.unicode_encode(file, "UTF-8"), label), num_parallel_calls=N_CORES)
    ds = ds.map(lambda file, label: (tf.strings.split(  # tf.strings.regex_replace(file, "\r", "")
        file, "\n"), label),
                num_parallel_calls=N_CORES)
    ds = ds.map(
        lambda lines, label: (tf.map_fn(fn=lambda line: tf.strings.unicode_decode(line, "UTF-8")[:max_len], elems=lines
                                        , fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32)),
                              label), num_parallel_calls=N_CORES)
    return ds


def process_snippet_ds(snippet_ds: tf.data.Dataset) -> tf.data.Dataset:
    MAX_LEN = 40
    snippet_ds = snippets_to_int_ds(snippet_ds, MAX_LEN)
    return preprocess_int_snippet(snippet_ds, MAX_LEN)


def get_snippet_ds(source_folder: str, shuffle: bool = False, label_mode: str = 'int') -> tf.data.Dataset:
    snippet_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory(
        source_folder, label_mode=label_mode, batch_size=None, shuffle=shuffle)
    return process_snippet_ds(snippet_ds)


if __name__ == '__main__':
    snippet_length: int = 5
    SNIPPET_SOURCE_FOLDER = ".\\processed_snippet5"
    snippet_ds: tf.data.Dataset = get_snippet_ds(SNIPPET_SOURCE_FOLDER)
    for item in snippet_ds.take(100):
        print(item)
