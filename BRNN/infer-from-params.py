#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example use of model inference, by restoring it from the saved model
"""
import pickle
import keras
from configuration import PICKLE_FILE_NAMES_700M, PICKLE_FILE_NAMES_400M
import os
from data import select_first_in_list
from utils import evaluate_model

MODELS_DIR = './models/'


def main():
    file_names = PICKLE_FILE_NAMES_700M

    #model_file_name = MODELS_DIR + file_names['model']
    model_names = [
        "RNN-700000000-batch_size_2048-n_rnn_layers_6-drop_out_0-n_neurons_hidden_dense_layer_classifier_512-n_class_layers_2-learning_rate_0.001-n_neurons_lstm_out_256-embedding_dim_32-activation_relu-lstm_True-",
        "RNN-700000000-batch_size_2048-n_rnn_layers_8-drop_out_0-n_neurons_hidden_dense_layer_classifier_512-n_class_layers_2-learning_rate_0.0005-n_neurons_lstm_out_256-embedding_dim_32-activation_relu-lstm_True-",
        "RNN-700000000-batch_size_2048-n_rnn_layers_10-drop_out_0-n_neurons_hidden_dense_layer_classifier_512-n_class_layers_2-learning_rate_0.0001-n_neurons_lstm_out_256-embedding_dim_32-activation_relu-lstm_True-"
        ]

    print("Loading X for validation...")
    with open(file_names['x_test'], 'rb') as handle:
        x_val = pickle.load(handle)
    print("Loading Y validation...")
    with open(file_names['y_test'], 'rb') as handle:
        y_val = pickle.load(handle)

    DATA_PCT = 1
    x_val, y_val = select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)

    for model_file_name in model_names:
        model_file_name = MODELS_DIR + model_file_name
        if not os.path.exists(model_file_name):
            print(f"File '{model_file_name}' not found.")
            return
        print(f"Loading model from '{model_file_name}'...")
        model = keras.models.load_model(model_file_name)
        print(f"Evaluating the model {model_file_name}...")

        evaluate_model(model, x_val, y_val)


if __name__ == "__main__":
    main()
