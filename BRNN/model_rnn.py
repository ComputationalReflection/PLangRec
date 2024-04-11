#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that models a bidirectional recurrent neural network (BRNN) for language classification
"""

import os
import pickle
from typing import List

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.layers import GRU
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer, LeakyReLU
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, Model
import keras
from tensorflow.keras import regularizers


def create_model(n_neurons_lstm_out: int, max_char_per_line: int, vocabulary_size: int,
                 embedding_dim, number_of_classes: int, dropout_factor: float,
                 n_neurons_hidden_dense_layer_classifier: int, n_rnn_layers: int, n_class_layers: int,
                 activation: str, lstm: bool):
    inputs = Input(shape=(max_char_per_line,))
    embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)
    x = embedding_layer(inputs)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    RNN_Class = LSTM if lstm else GRU
    for layer_index in range(n_rnn_layers-1):
        #x = Bidirectional(RNN_Class(n_neurons_lstm_out, dropout=dropout_factor, return_sequences=True,  \
        #                           kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2)))(x)
        #x = Bidirectional(RNN_Class(n_neurons_lstm_out, dropout=dropout_factor, recurrent_dropout=dropout_factor, return_sequences=True))(x)
        x = Bidirectional(RNN_Class(n_neurons_lstm_out, dropout=dropout_factor, return_sequences=True))(x)   # ñapa para reducir el impacto del dropout (se queda sin memoria)
    #x = Bidirectional(RNN_Class(n_neurons_lstm_out, dropout=dropout_factor, kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2)))(x)
    x = Bidirectional(RNN_Class(n_neurons_lstm_out, dropout=dropout_factor, recurrent_dropout=dropout_factor))(x)
    for layer_index in range(n_class_layers):
        if activation == 'leaky_relu':
            x = Dense(n_neurons_hidden_dense_layer_classifier)(x)
            x = LeakyReLU()(x)
        else:
            x = Dense(n_neurons_hidden_dense_layer_classifier, activation=activation)(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    outputs = Dense(number_of_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

