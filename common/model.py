from io import BytesIO
from typing import Dict, Tuple, List
import requests
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from keras import Model
import os
import sys
import tensorflow_datasets as tfds


MODEL_PATH = '../common/model/BRNN'
MODEL_URL = 'https://reflection.uniovi.es/bigcode/download/2024/plangrec/BRNN.zip'
META_MODEL_PATH = '../common/meta-model'
META_MODEL_URL = 'https://reflection.uniovi.es/bigcode/download/2024/plangrec/ensemble-meta-model.zip'
META_MODEL_FILE_NAME = 'ensemble-meta-model.h5'

N_CORES = tf.data.AUTOTUNE
NUMBER_OF_BINS = 100
NUMBER_OF_LANGUAGES = 21


def _download_and_load_model(model_path: str, model_url: str, file_name: str = None) -> Model:
    if not os.path.exists(model_path):
        print(f"Model not found in '{model_path}'.")
        print(f"Downloading model from '{model_url}'. It may take some minutes...")
        response = requests.get(model_url)
        if response.status_code == 200:
            # Extract the zip file content
            with ZipFile(BytesIO(response.content)) as zip_file:
                # Create the directory for extraction if it doesn't exist
                os.makedirs(model_path, exist_ok=True)
                # Extract all contents to the specified path
                zip_file.extractall(model_path)
                print(f"Model successfully downloaded and extracted to '{model_path}'.\n")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            sys.exit(-1)
    if file_name:
        return tf.keras.models.load_model(os.path.join(model_path, file_name))
    else:
        return tf.keras.models.load_model(model_path)


# Loads the model and meta-model into memory at startup to go faster upon prediction
model = _download_and_load_model(MODEL_PATH, MODEL_URL)
meta_model = _download_and_load_model(META_MODEL_PATH, META_MODEL_URL, META_MODEL_FILE_NAME)

def _parse_line(line, allow_short_lines: bool):
    line = line.strip()  # Remove blank at the beginning and at the end
    if len(line) < 10 and not allow_short_lines:
        return None  # The line will not be considered (too short)
    result = list(line)  # Split line in characters
    # Tokenization
    result = result[:40]  # Truncates line
    result = [ord(char) for char in result]  # Parse characters to its ASCII code
    # Vocabulary
    result = [code if (32 <= code < 127) else 31 for code in result]  # Filter vocabulary
    result = [code-30 for code in result]  # Makes a consecutive vocabulary
    if len(result) < 40:
        result.extend([0] * (40 - len(result)))  # Padding
    return result


def _preprocess_char_numbers(ds):
    empty_spaces = 32 - 2
    ds = ds.map(lambda line: (line - empty_spaces), num_parallel_calls=2)
    out_of_vocabulary = 1
    ds = ds.map(lambda line: (tf.where(line < 32 - empty_spaces, out_of_vocabulary, line)), num_parallel_calls=2)
    ds = ds.map(lambda line: (tf.where(line >= 127 - empty_spaces, out_of_vocabulary, line)), num_parallel_calls=2)
    return ds


def _preprocess_numeric_ds(ds):
    ds = ds.map(lambda line: line[:40])
    ds = _preprocess_char_numbers(ds)
    ds = ds.map(lambda line: (tf.concat([line, tf.zeros([40 - tf.shape(line)[0]], dtype=tf.dtypes.int32)], axis=0)),
                num_parallel_calls=2)
    return ds


def _soft_voting(predictions):
    # Compute the average values of the columns
    column_averages = np.mean(predictions, axis=0)
    return column_averages / np.sum(column_averages)


def _snippets_to_int_ds(ds: tf.data.Dataset, max_len: int):
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

def _preprocess_int_snippet(ds: tf.data.Dataset, max_len: int):
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


def _process_snippet_ds(snippet_ds: tf.data.Dataset) -> tf.data.Dataset:
    MAX_LEN = 40
    snippet_ds = _snippets_to_int_ds(snippet_ds, MAX_LEN)
    return _preprocess_int_snippet(snippet_ds, MAX_LEN)

def _n_lines_2_frequencies(predictions: np.ndarray, number_of_bins: int) -> np.ndarray:
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

def _predict_by_parts(model: tf.keras.Model, X: np.ndarray, batch_size: int) -> np.ndarray:
    probY = []
    split_size: int = 10_000_000
    parts = list(range(0, X.shape[0], split_size))
    parts.append(X.shape[0])
    for i in range(len(parts) - 1):
        probY.append(model.predict(X[parts[i]:parts[i + 1]], batch_size=batch_size))
    return np.concatenate(probY, axis=0)

def _trainX_of_ragged_text_to_lines(ragged_array):
    numpy_arrays = []
    # Convert each RaggedTensor to a numpy array
    for ragged_tensor in ragged_array:
        dense_tensor = ragged_tensor.to_tensor()
        numpy_array = dense_tensor.numpy()
        numpy_arrays.append(numpy_array)
    # Concatenate all the numpy arrays into a single numpy array
    return np.concatenate(numpy_arrays, axis=0)

def _frequency_vectors_from_files_dir(train_file_ds: tf.data.Dataset, single_line_model: tf.keras.Model,
                                      number_of_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    train_file_ds = train_file_ds.map(lambda snippet, label: (snippet, tf.shape(snippet)[0], label))
    numpy_ds = np.fromiter(tfds.as_numpy(train_file_ds), dtype=(object, 3))
    trainX = numpy_ds[:, 0]
    snippet_lengths = numpy_ds[:, 1]
    real_Y = np.stack(numpy_ds[:, 2])
    trainX = _trainX_of_ragged_text_to_lines(trainX)
    PREDICT_BATCH = 2_048
    probY = _predict_by_parts(single_line_model, trainX, PREDICT_BATCH)
    current_index: int = 0
    X_train_meta_temp: List[np.ndarray] = []
    for length in snippet_lengths:
        frequencies = _n_lines_2_frequencies(probY[current_index:current_index + length], number_of_bins)
        X_train_meta_temp.append(frequencies)
        current_index += length

    return np.array(X_train_meta_temp), real_Y


def _predict_ensemble(snippet: str, single_line_model: tf.keras.Model, ensemble_model: tf.keras.Model) -> np.ndarray:
    """Predict the programming language of a given code snippet using an ensemble meta-model and a model."""
    snippet = "\n".join([line.strip() for line in snippet.split("\n")]).encode('utf-16le')
    snippet_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([snippet]).map(
        lambda snippet: (snippet, tf.zeros(shape=(21,))))
    snippet_ds = _process_snippet_ds(snippet_ds)
    frequency_vector, _ = _frequency_vectors_from_files_dir(snippet_ds, single_line_model, NUMBER_OF_BINS)
    frequency_vector = frequency_vector.reshape(frequency_vector.shape[0], -1)
    prediction = ensemble_model.predict(frequency_vector)
    return prediction


def predict(source_code: str) -> Dict[str, float]:
    from languages import LANGUAGES
    global model
    lines = source_code.split('\n')
    result = {}
    parsed_lines = [_parse_line(line, allow_short_lines=False) for line in lines
                    if _parse_line(line, allow_short_lines=False)]
    if len(parsed_lines) == 0:  # If no line is longer than 10, we take the first one
        parsed_lines = [_parse_line(lines[0], allow_short_lines=True)]
    # prediction calling the model if it is just one line
    if len(parsed_lines) == 1:
        single_prediction = model.predict(parsed_lines)[0]
    else:  # use the meta-model
        single_prediction = _predict_ensemble(source_code, model, meta_model)[0]
    for i, p in enumerate(single_prediction):
        result[LANGUAGES[i]] = round(p*100, 2)
    return result

