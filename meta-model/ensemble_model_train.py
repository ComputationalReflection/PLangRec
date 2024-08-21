#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module creates and trains a Multilayer Perceptron (MLP) ensemble model for multi-line language identification
trained with datasets of aggregated single-line predictions. The module includes functions for defining the MLP
architecture with customizable hyperparameters such as the optimizer, number of layers, units per layer,
activation function, learning rate and batch size. It loads the training set from pickle files and trains the MLP model,
reserving the 10% of the data for validation.
Finally, it saves to disk the best trained model selected with an Early Stopping callback, and prints the corresponding
validation loss and accuracy. This allows for flexible hyperparameter tuning with different MLP configurations
to achieve optimal performance.
"""

from ensemble_model_generate_dataset import NUMBER_OF_BINS, NUMBER_OF_LANGUAGES
import tensorflow as tf
from pickle_load import pickle_to_numpy
import sys
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization


def create_train_meta_model(X_train_meta, y_train_meta, number_of_bins: int, lerning_rate: float,
                            is_adam: bool, batch_size, n_layers: int, n_hidden: int, activation: str):
    # X_train_meta = (SAMPLES, NUMBER_OF_LANGS, BINS)
    # y_train_meta = (SAMPLES, NUMBER_OF_LANGS)
    # Flatten the X_train_meta dataset to pass it to a model
    X_train_meta_flattened = X_train_meta.reshape(X_train_meta.shape[0], -1)
    meta_model = Sequential()
    for _ in range(n_layers):
        if activation != 'selu':
            meta_model.add(
                BatchNormalization(input_dim=NUMBER_OF_LANGUAGES * number_of_bins))  # Adding BatchNormalization layer
        meta_model.add(Dense(n_hidden, activation=activation))
    meta_model.add(Dense(NUMBER_OF_LANGUAGES, activation='softmax'))

    # Train the meta-model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lerning_rate) if is_adam else tf.keras.optimizers.SGD(
        learning_rate=lerning_rate, momentum=0.9, nesterov=True)
    meta_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    hist = meta_model.fit(X_train_meta_flattened, y_train_meta, epochs=20, batch_size=batch_size, validation_split=0.1,
                          callbacks=[early_stopping_callback])
    return meta_model, hist


def train_and_save_model(X_train, y_train, learning_rate, is_adam, batch_size, n_layers, n_hidden, activation: str):
    print(
        f"Training meta-model with Adam: {is_adam}, learning rate: {learning_rate}, batch_size {batch_size}, n_layers {n_layers}, n_hidden {n_hidden}, activation {activation}")
    ini_time = time.time()
    model, history = create_train_meta_model(X_train, y_train, NUMBER_OF_BINS, lerning_rate=learning_rate,
                                             is_adam=is_adam, batch_size=batch_size, n_layers=n_layers,
                                             n_hidden=n_hidden, activation=activation)
    print(f"Time to create and train meta-model: {time.time() - ini_time}")
    model.save(
        f"model_{sys.argv[1]}_adam{is_adam}_lr{learning_rate}_batch{batch_size}_nlayers{n_layers}_nhidden{n_hidden}_activation{activation}.h5")
    min_loss = min(history.history['val_loss'])
    return min_loss, history.history['val_accuracy'][history.history['val_loss'].index(min_loss)]


if __name__ == '__main__':
    ini_time = time.time()
    trainX = pickle_to_numpy(f"X_train_{sys.argv[1]}")
    trainY = pickle_to_numpy(f"y_train_{sys.argv[1]}")
    load_time = time.time()
    print(f"Time to load pickles: {load_time - ini_time}")
    learning_rate = 0.01
    batch_size = 128
    is_adam = False
    n_hidden = 300
    n_layers = 1
    activation = 'relu'
    loss, acc = train_and_save_model(trainX, trainY, learning_rate, is_adam, batch_size, n_layers, n_hidden, activation)
    print(
        f"Loss for Adam: {is_adam}, learning rate: {learning_rate}, batch_size {batch_size}, n_layers {n_layers}, n_hidden {n_hidden}, activation {activation} is {loss} and acc {acc}")
