#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that receives a model and source code and predicts the probability for all the languages.
If the code has more than one line, a soft voting ensemble approach is performed.
"""
from typing import Dict
import numpy as np
import tensorflow as tf
from keras import Model
from configuration import LANGUAGES


def _parse_line(line, allow_short_lines: bool):
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


def predict_from_code(model: Model, source_code: str) -> Dict[str, float]:
    lines = source_code.split('\n')
    result = {}
    parsed_lines = [_parse_line(line, allow_short_lines=False) for line in lines
                    if _parse_line(line, allow_short_lines=False)]
    if len(parsed_lines) == 0:  # If no line is longer than 10, we take the first one
        parsed_lines = [_parse_line(lines[0], allow_short_lines=True)]
    predictions = model.predict(parsed_lines)
    single_prediction = _soft_voting(predictions)
    for i, p in enumerate(single_prediction):
        result[LANGUAGES[i]] = round(p*100, 2)
    return result

