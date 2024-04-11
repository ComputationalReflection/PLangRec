#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performs an error analysis, showing the intances that were missclassified
"""

import pickle
import keras
import numpy as np
from configuration import PICKLE_FILE_NAMES_400M

MODEL_FILE_NAME = "models/RNN-432180483-batch_size_2048-n_rnn_layers_8-drop_out_0-n_neurons_hidden_dense_layer_classifier_512-n_class_layers_2-learning_rate_0.0001-n_neurons_lstm_out_256-embedding_dim_32-activation_relu-lstm_True-"
LANGUAGE_LABELS = ["Assembly", "C", "C++", "C#", "CSS", "Go", "HTML", "Java", "JavaScript", "Kotlin",
                   "Matlab", "Perl", "PHP", "Python", "R", "Ruby", "Scala", "SQL", "Swift", "TypeScript",
                   "Unix Shell"]
NUMBER_INSTANCES_TO_PROCESS = 100_000


def load_dataset(x_file_name: str, y_file_name: str):
    print("Loading X for validation...")
    with open(x_file_name, 'rb') as handle:
        x_test = pickle.load(handle)
    print("Loading Y validation...")
    with open(y_file_name, 'rb') as handle:
        y_test = pickle.load(handle)
    # Convert from one-hot to integer values
    y_test = np.argmax(y_test, axis=1)
    print(x_test.shape)
    print(y_test.shape)
    return x_test, y_test


def shuffle_dataset(x_data, y_data):
    print("Shuffling dataset...")
    assert len(x_data) == len(y_data)
    indices = np.random.permutation(len(x_data))
    return x_data[indices], y_data[indices]


def get_erroneous_predictions(y_test, predicted_y, actual_lang: str, predicted_lang: str):
    assert len(y_test) == len(predicted_y)
    actual_lang_index, predicted_lang_index = LANGUAGE_LABELS.index(actual_lang), LANGUAGE_LABELS.index(predicted_lang)
    wrong_classification_indexes = [i for i in range(len(y_test)) if y_test[i] == actual_lang_index and
                                    np.argmax(predicted_y[i]) == predicted_lang_index]
    return wrong_classification_indexes


def convert_vector_to_code(vector) -> str:
    return "".join(map(lambda integer: chr(integer - 2 + 32), vector))


def show_error_instances(wrong_classification_indexes, x_test):
    for index in wrong_classification_indexes:
        print(f'Source code line: "{convert_vector_to_code(x_test[index])}".')


def show_miss_classifications(actual_lang: str, predicted_lang: str):
    # Load dataset
    x_test, y_test = load_dataset(PICKLE_FILE_NAMES_400M['x_test'], PICKLE_FILE_NAMES_400M['y_test'])
    x_test, y_test = shuffle_dataset(x_test, y_test)
    x_test, y_test = x_test[:NUMBER_INSTANCES_TO_PROCESS], y_test[:NUMBER_INSTANCES_TO_PROCESS]
    # Load model
    print(f"Loading the model from {MODEL_FILE_NAME} ...")
    model = keras.models.load_model(MODEL_FILE_NAME)
    # Predict the languages
    predicted_y = model.predict(x_test)
    wrong_classification_indexes = get_erroneous_predictions(y_test, predicted_y, actual_lang, predicted_lang)
    print(f"Miss classifications of {actual_lang} (actual) as {predicted_lang} (predicted):")
    show_error_instances(wrong_classification_indexes, x_test)


def main():
    show_miss_classifications("Swift", "Kotlin")


if __name__ == "__main__":
    main()
