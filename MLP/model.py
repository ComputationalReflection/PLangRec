#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is devoted to compile a Multilayer Perceptron (MLP) model using TensorFlow's Keras API. The
compile_model function constructs the model architecture based on various hyperparameters such as number of dense
layers, embedding size, number of neurons per layer, activation function, dropout rate, initializers, and learning
rate. It also takes into account parameters regarding the shape of the dataset such as line length, vocabulary size
and number of labels. It builds the model by adding hidden layers with specified activation functions and
initializers, including batch normalization layers if the SELU activation is not chosen. Finally, it compiles the
model with categorical crossentropy loss, Nesterov Accelerated Gradient (NAG), and evaluates metrics including
accuracy, precision, and recall. """

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Embedding, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from parameters import SELU_ACTIVATION, LEAKY_RELU_ACTIVATION


def add_hidden_layer_to_model(model: Sequential, activation: str, initializer: str, denses_width: int) -> None:
    if activation == LEAKY_RELU_ACTIVATION:
        model.add(Dense(units=denses_width, kernel_initializer=initializer))
        model.add(LeakyReLU(alpha=0.1))
    else:
        model.add(Dense(units=denses_width, activation=activation, kernel_initializer=initializer))
    if activation != SELU_ACTIVATION:
        model.add(BatchNormalization())


def compile_model(max_len: int, vocab_size: int, n_labels: int, n_denses: int,
                  denses_width: int, embedding_size: int, flatten_down: bool,
                  dropout: float, activation: str, initializer: str, lr: float) -> Sequential:
    model = Sequential()
    if embedding_size is None:
        model.add(Input(shape=(max_len * vocab_size,)))
    else:
        model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
        if flatten_down:
            model.add(Flatten(input_shape=[max_len * embedding_size]))
        if activation != SELU_ACTIVATION:
            model.add(BatchNormalization())

    if n_denses >= 1:
        add_hidden_layer_to_model(model, activation, initializer, denses_width)
        for _ in range(n_denses - 1):
            if dropout > 0.0:
                model.add(Dropout(rate=dropout))
            add_hidden_layer_to_model(model, activation, initializer, denses_width)

    if not flatten_down:
        model.add(Flatten(input_shape=[embedding_size * denses_width]))

    model.add(Dense(n_labels, activation='softmax'))
    lr = tf.Variable(lr, trainable=False, dtype=tf.float32)
    optimizer = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', Precision(name="precision"), Recall(name="recall")])
    print(model.summary())

    return model
