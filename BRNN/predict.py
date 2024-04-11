#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example use of model inference that restores it from the saved model, receives and input code and prints
the classification probabilities for the all the languages.
"""

import keras
import os
from perdict_code import predict_from_code

MODELS_DIR = './models/'
MODEL_NAME = ('RNN-432180483-batch_size_2048-n_rnn_layers_8-drop_out_0-n_neurons_hidden_dense_layer_classifier_512-'
              'n_class_layers_2-learning_rate_0.0001-n_neurons_lstm_out_256-embedding_dim_32-activation_relu-'
              'lstm_True-')


def predict(source_code: str) -> None:
    model_file_name = os.path.join(MODELS_DIR, MODEL_NAME)
    if not os.path.exists(model_file_name):
        print(f"File '{model_file_name}' not found.")
        return
    print(f"Loading model from '{model_file_name}'...")
    model = keras.models.load_model(model_file_name)
    print(f"Predicting the source code:\n\t{source_code}")
    results = predict_from_code(model, source_code)
    print(f"\nPredictions:\n\t{results}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print('Usage: python predict.py "<source code>"')
        sys.exit(-1)
    predict(sys.argv[1])

