#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module performs inference to predict the programming language of a given code
snippet of any length. The code snippet is processed into a format suitable for the model,
including converting it into single-line numeric vectors and aggregating predictions
from a BRNN model into a frequency histogram. Then, an MLP ensemble model predicts the language probabilities
based on this histogram. The predicted probabilities for each language are returned,
indicating the likelihood of the snippet belonging to each supported programming
language. This module ensures that code snippets of varying lengths can be accurately
analyzed and classified.
"""

import tensorflow as tf
from snippet_to_preprocessed import process_snippet_ds
from ensemble_model_generate_dataset import frequency_vectors_from_files_dir, NUMBER_OF_BINS
import numpy as np

LANG_LIST = ['asm', 'c', 'cpp', 'csharp', 'css', 'go', 'html', 'java', 'js', 'kotlin', 'matlab', 'perl', 'php',
             'python', 'r', 'ruby', 'scala', 'sql', 'swift', 'ts', 'unix']


def inference(snippet: str, single_line_model: tf.keras.Model, ensemble_model: tf.keras.Model) -> np.ndarray:
    snippet = "\n".join([line.strip() for line in snippet.split("\n")]).encode('utf-16le')
    snippet_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([snippet]).map(
        lambda snippet: (snippet, tf.zeros(shape=(21,))))
    snippet_ds = process_snippet_ds(snippet_ds)
    frequency_vector, _ = frequency_vectors_from_files_dir(snippet_ds, single_line_model, NUMBER_OF_BINS)
    frequency_vector = frequency_vector.reshape(frequency_vector.shape[0], -1)
    prediction = ensemble_model.predict(frequency_vector)
    return prediction


if __name__ == "__main__":
    BRNN_MODEL_PATH: str = f".\\models\\layers8"
    ENSEMBLE_MODEL_PATH: str = f".\\models\\ensemble_model.h5"

    brnn_model = tf.keras.models.load_model(BRNN_MODEL_PATH)
    ensemble_model = tf.keras.models.load_model(ENSEMBLE_MODEL_PATH)
    JAVA_LINE = "   public class Person implements Serializable{ \n//One line comment Person\n\t private int age;\n\t private String name;\t\n/*Multiline comment \nfor the people with age and name\n*/\n}"
    prediction = inference(JAVA_LINE, brnn_model, ensemble_model)
    print(LANG_LIST[np.argmax(prediction[0])])
    SHELL_LINE = "echo \"Hello, World!\"\n#Single line comment about these commands\n ls -la dir > file.txt\n sudo chmod 777 another_file"
    prediction = inference(SHELL_LINE, brnn_model, ensemble_model)
    print(LANG_LIST[np.argmax(prediction[0])])

