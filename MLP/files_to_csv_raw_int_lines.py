#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This module transforms text files into integer vectors mapping each character to its UTF-8 value and stores them as CSV
files. It reads the texts in the corpus and splits the different lines, concatenates them with their corresponding
labels, and then batches them for efficient processing. It subsequently iterates through the resulting dataset,
writing batches of lines into separate CSV files. Additionally, it also handles the dataset stratification."""

import tensorflow as tf
from typing import List
import os

N_CORES = tf.data.AUTOTUNE


def eff_write(ds, folder, file_counter=0, lines_per_file=1000):
    if not os.path.exists(folder):
        os.mkdir(folder)
    ds = ds.map(lambda line, label: tf.strings.reduce_join(tf.strings.as_string(tf.concat([line, [label]], axis=0))
                                                           , separator=","))
    ds = ds.batch(lines_per_file)
    FILE_NAME = "f"
    for batch in ds:
        complete_name = FILE_NAME + str(file_counter) + ".csv"
        file_content = str(tf.strings.reduce_join(batch, separator="\n").numpy(), encoding="ascii")
        with open(folder + "/" + complete_name, 'w') as csvfile:
            csvfile.write(file_content)
        file_counter += 1
    return file_counter


def line_to_raw_int_ds(ds: tf.data.Dataset):
    ds = ds.map(lambda file, label: (file, tf.cast(label, tf.dtypes.int32)), num_parallel_calls=N_CORES)
    ds = ds.map(lambda file, label: (tf.strings.unicode_decode(file, "UTF-16LE"), label), num_parallel_calls=N_CORES)
    ds = ds.map(lambda file, label: (tf.strings.unicode_encode(file, "UTF-8"), label), num_parallel_calls=N_CORES)
    ds = ds.interleave(lambda file, label: tf.data.Dataset.from_tensor_slices(
        tf.map_fn(lambda line: (line, label), tf.strings.split(tf.strings.regex_replace(file, "\r", ""), "\n"),
                  fn_output_signature=(tf.dtypes.string, tf.int32))).shuffle(1000)
                       , num_parallel_calls=N_CORES,
                       deterministic=True, block_length=1, cycle_length=20_000
                       )
    LENGTH_LIMIT: int = 10
    ds = ds.filter(lambda line, _: tf.strings.length(line) >= LENGTH_LIMIT)
    return ds.map(lambda line, label: (tf.strings.unicode_decode(line, "UTF-8"), label), num_parallel_calls=N_CORES)


def get_raw_lines_ds(source_folder: str) -> tf.data.Dataset:
    snippet_ds: tf.data.Dataset = tf.keras.utils.text_dataset_from_directory(
        source_folder, label_mode='int', batch_size=None, shuffle=True)
    return line_to_raw_int_ds(snippet_ds)


def stratify_ds(ds: tf.data.Dataset, weights: List[float]):
    datasets = [ds.filter(lambda _, label: label == i) for i in range(len(weights))]
    return tf.data.Dataset.sample_from_datasets(datasets, weights, stop_on_empty_dataset=True)


if __name__ == '__main__':
    SNIPPET_SOURCE_FOLDER = ".\\comments_V2_TXT_test"
    line_ds: tf.data.Dataset = get_raw_lines_ds(SNIPPET_SOURCE_FOLDER)
    DS_SIZE: int = 1_000_000
    N_LANGS: int = 21
    WEIGHTS = [DS_SIZE / N_LANGS] * N_LANGS
    line_ds = stratify_ds(line_ds, WEIGHTS).take(DS_SIZE)
    DEST_FOLDER: str = ".\\raw_int_lines_test_balanced_1M"
    eff_write(line_ds, DEST_FOLDER)
