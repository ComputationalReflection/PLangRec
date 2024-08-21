#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides functionality for generating datasets by aggregating single-line predictions from variable-length
code snippets into fixed-size histogram vectors to train multi-line ensemble models. It loads text
snippets from files into a TensorFlow dataset, processes and transforms each line into a numeric vector, and loads
a keras model to predict language probabilities for each line.
These single-line predictions are aggregated into frequency histograms, representing the relative distribution of language
probabilities across multiple lines. The resulting feature matrices and corresponding labels are stored in pickle files
for future training of ensemble models.
In this way, it enables the conversion of texts of varying lengths into fixed-size vectors, facilitating the training
of robust multi-line language detection models.
"""

import numpy
import tensorflow as tf
import tensorflow_datasets as tfds
from snippet_to_preprocessed import get_snippet_ds
import numpy as np
from typing import List, Tuple
import time
from pickle_store import numpy_to_pickle
import sys


def n_lines_2_frequencies(predictions: np.ndarray, number_of_bins: int) -> np.ndarray:
    # Notice: number_of_lines could be variable, so we reduce predictions to
    # (NUMBER_OF_LANGUAGES * number_of_lines)
    frequencies = np.zeros((NUMBER_OF_LANGUAGES, number_of_bins))
    for i in range(NUMBER_OF_LANGUAGES):
        # Get the i column of predictions (language i) and create a histogram with NUMBER_OF_BINS bins
        hist, _ = np.histogram(predictions[:, i], bins=number_of_bins, range=(0, 1))
        frequencies[i] += hist
    number_of_lines = predictions.shape[0]
    frequencies = frequencies / (number_of_lines * NUMBER_OF_LANGUAGES)
    return frequencies


def trainX_of_ragged_text_to_lines(ragged_array):
    numpy_arrays = []

    # Convert each RaggedTensor to a numpy array
    for ragged_tensor in ragged_array:
        dense_tensor = ragged_tensor.to_tensor()
        numpy_array = dense_tensor.numpy()
        numpy_arrays.append(numpy_array)

    # Concatenate all the numpy arrays into a single numpy array
    return np.concatenate(numpy_arrays, axis=0)


def predict_by_parts(model: tf.keras.Model, X: np.ndarray, batch_size: int) -> np.ndarray:
    probY = []
    split_size: int = 10_000_000
    parts = list(range(0, X.shape[0], split_size))
    parts.append(X.shape[0])
    for i in range(len(parts) - 1):
        probY.append(model.predict(X[parts[i]:parts[i + 1]], batch_size=batch_size))
    return np.concatenate(probY, axis=0)


def frequency_vectors_from_files_dir(train_file_ds: tf.data.Dataset, single_line_model: tf.keras.Model, number_of_bins: int) -> Tuple[
    numpy.ndarray, numpy.ndarray]:
    train_file_ds = train_file_ds.map(lambda snippet, label: (snippet, tf.shape(snippet)[0], label))
    numpy_ds = np.fromiter(tfds.as_numpy(train_file_ds), dtype=(object, 3))

    trainX = numpy_ds[:, 0]
    snippet_lengths = numpy_ds[:, 1]
    real_Y = np.stack(numpy_ds[:, 2])

    trainX = trainX_of_ragged_text_to_lines(trainX)

    PREDICT_BATCH = 2_048
    probY = predict_by_parts(single_line_model, trainX, PREDICT_BATCH)
    current_index: int = 0
    X_train_meta_temp: List[numpy.ndarray] = []
    for length in snippet_lengths:
        frequencies = n_lines_2_frequencies(probY[current_index:current_index + length], number_of_bins)
        X_train_meta_temp.append(frequencies)
        current_index += length

    return np.array(X_train_meta_temp), real_Y


NUMBER_OF_BINS = 100
NUMBER_OF_LANGUAGES = 21

if __name__ == '__main__':
    BRNN_MODEL_PATH: str = f".\\models\\layers8"
    SNIPPET_FILE_FOLDER = f".\\{sys.argv[1]}"

    ini_time = time.time()
    train_file_ds: tf.data.Dataset = get_snippet_ds(SNIPPET_FILE_FOLDER, shuffle=True, label_mode='categorical')
    load_ds_time = time.time()
    print("Time to load dataset: {}".format(load_ds_time - ini_time))
    X_train_meta, y_train_meta = frequency_vectors_from_files_dir(train_file_ds, tf.keras.models.load_model(BRNN_MODEL_PATH), NUMBER_OF_BINS)
    extract_time = time.time()
    print("Whole time to extract matrices: {}".format(extract_time - load_ds_time))
    description: str = f"{X_train_meta.shape}_" + str(BRNN_MODEL_PATH.split("\\")[-1])
    numpy_to_pickle(X_train_meta, f"X_train_files_{description}")
    numpy_to_pickle(y_train_meta, f"y_train_files_{description}")
