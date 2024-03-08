from io import BytesIO
from typing import Dict
import requests
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from keras import Model

MODEL_PATH = '../common/model/BRNN'
MODEL_URL = 'https://reflection.uniovi.es/bigcode/download/2024/plangrec/BRNN.zip'


def download_and_load_model(model_path: str, model_url: str) -> Model:
    import os
    import sys
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
    return tf.keras.models.load_model(model_path)


# Loads the model into memory at startup to go faster upon prediction
model = download_and_load_model(MODEL_PATH, MODEL_URL)


def parse_line(line, allow_short_lines: bool):
    line = line.strip()  # Remove blank at the beginning and at the end
    if len(line) < 10 and not allow_short_lines:
        return None  # The line will not be considered (to short)
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


def preprocess_char_numbers(ds):
    empty_spaces = 32 - 2
    ds = ds.map(lambda line: (line - empty_spaces), num_parallel_calls=2)
    out_of_vocabulary = 1
    ds = ds.map(lambda line: (tf.where(line < 32 - empty_spaces, out_of_vocabulary, line)), num_parallel_calls=2)
    ds = ds.map(lambda line: (tf.where(line >= 127 - empty_spaces, out_of_vocabulary, line)), num_parallel_calls=2)
    return ds


def preprocess_numeric_ds(ds):
    ds = ds.map(lambda line: line[:40])
    ds = preprocess_char_numbers(ds)
    ds = ds.map(lambda line: (tf.concat([line, tf.zeros([40 - tf.shape(line)[0]], dtype=tf.dtypes.int32)], axis=0)),
                num_parallel_calls=2)
    return ds


def soft_voting(predictions):
    # Compute the average values of the columns
    column_averages = np.mean(predictions, axis=0)
    return column_averages / np.sum(column_averages)


def predict(source_code: str) -> Dict[str, float]:
    from languages import LANGUAGES
    global model
    lines = source_code.split('\n')
    result = {}
    parsed_lines = [parse_line(line, allow_short_lines=False) for line in lines
                    if parse_line(line, allow_short_lines=False)]
    if len(parsed_lines) == 0:  # If no line is longer than 10, we take the first one
        parsed_lines = [parse_line(lines[0], allow_short_lines=True)]
    predictions = model.predict(parsed_lines)
    single_prediction = soft_voting(predictions)
    for i, p in enumerate(single_prediction):
        result[LANGUAGES[i]] = round(p*100, 2)
    return result

