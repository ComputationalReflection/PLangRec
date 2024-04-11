#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that trains a bidirectional recurrent neural network (BRNN) for language classification
https://blog.paperspace.com/transformers-text-classification/
"""

import tensorflow as tf
import random

from configuration import PICKLE_FILE_NAMES_70M, DATA_PCT, \
    MAX_CHARS_PER_LINE, VOCABULARY_SIZE, NUMBER_OF_CLASSES, EMBEDDING_DIM, \
    PICKLE_FILE_NAMES_700M, MODELS_DIR, PICKLE_FILE_NAMES_7M, PICKLE_FILE_NAMES_400M, LOG_DIR, CSV_DIR
from data import load_data, select_first_in_list
from hyperparams import HyperParams
from model_rnn import create_model
from utils import compile_model, train_model, evaluate_model, train_model_lazy, \
    save_model, f1_m, precision_m, recall_m, disable_keras_warning_messages, create_log_file


def main_test():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_70M)
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                     select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)
    model = create_model(n_neurons_lstm_out= MAX_CHARS_PER_LINE, max_char_per_line=MAX_CHARS_PER_LINE,
                         vocabulary_size=VOCABULARY_SIZE,
                         embedding_dim=EMBEDDING_DIM, number_of_classes=NUMBER_OF_CLASSES, dropout_factor=0,
                         n_neurons_hidden_classifier=512)
    model.summary()
    compile_model(model, "adam", "categorical_crossentropy", ["accuracy"])
    history = train_model(model, 'val_accuracy', patience=0, verbose=1, batch_size=64, epochs=1,
                          x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                          file_name="output.txt")
    # ------------ Model evaluation ------------------
    evaluate_model(model, x_val, y_val)


def main_train():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_7M)
    DATA_PCT = 0.001
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)
    hyper = HyperParams("RNN", x_train.shape[0], batch_size=2048,
                        n_rnn_layers=1,
                        n_neurons_lstm_out=128, drop_out=0,
                        n_neurons_hidden_dense_layer_classifier=512,
                        n_class_layers=1)
    model = create_model(n_neurons_lstm_out=hyper.get('n_neurons_lstm_out'),
                         max_char_per_line=MAX_CHARS_PER_LINE,
                         vocabulary_size=VOCABULARY_SIZE,
                         embedding_dim=EMBEDDING_DIM, number_of_classes=NUMBER_OF_CLASSES,
                         n_class_layers=hyper.get('n_class_layers'),
                         dropout_factor=hyper.get('drop_out'),
                         n_rnn_layers=hyper.get('n_rnn_layers'),
                         n_neurons_hidden_dense_layer_classifier=hyper.get('n_neurons_hidden_dense_layer_classifier'))
    model.summary()
    model_name = hyper.long_name()
    print(model_name)

    compile_model(model, "adam", "categorical_crossentropy", ["accuracy"])
    history = train_model_lazy(model, patience=0, verbose=2, batch_size=hyper.get('batch_size'), epochs=50,
                               x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                               early_stop_monitor="val_accuracy", model_file_name=model_name,
                               log_file_name=LOG_DIR + model_name, csv_file_name= CSV_DIR + model_name, patience_lr=0)
    # ------------ Model evaluation ------------------
    evaluate_model(model, x_val, y_val)


def main_stochastic():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_7M)
    DATA_PCT = 0.01
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)

    for iteration in range(1000):
        print(f"Iteration: {iteration+1}.")
        # Random hyperparams
        n_rnn_layers = random.choice(range(1, 6))  # 1 - 5
        n_class_layers = random.choice(range(1, 3))  # 1 - 2
        n_neurons_lstm_out = random.choice([2**n for n in range(5, 11+1)])  # 32 - 2048
        n_neurons_hidden_dense_layer_classifier = random.choice([2**n for n in range(5, 11+1)])  # 32 - 2048
        embedding_dim = random.choice([8*n for n in range(1, 5+1)])  # 8, 16, 24, 32, 64
        learning_rate = random.choice([1*(10**n) for n in range(-6, 1)])  # 0.000_001 - 1
        batch_size = random.choice([2**n for n in range(5, 11+1)])  # 32 - 2048
        drop_out = random.choice([0, 0.1, 0.2])
        activation = random.choice(["relu", "selu", "elu"])

        hyper = HyperParams("RNN", x_train.shape[0],
                            n_rnn_layers=n_rnn_layers,
                            n_class_layers=n_class_layers,
                            n_neurons_lstm_out=n_neurons_lstm_out,
                            n_neurons_hidden_dense_layer_classifier=n_neurons_hidden_dense_layer_classifier,
                            embedding_dim=embedding_dim,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            drop_out=drop_out,
                            activation=activation,
                            )
        model = create_model(max_char_per_line=MAX_CHARS_PER_LINE,
                             vocabulary_size=VOCABULARY_SIZE,
                             number_of_classes=NUMBER_OF_CLASSES,
                             n_rnn_layers=hyper.get('n_rnn_layers'),
                             n_class_layers=hyper.get('n_class_layers'),
                             n_neurons_lstm_out=hyper.get('n_neurons_lstm_out'),
                             n_neurons_hidden_dense_layer_classifier=hyper.get('n_neurons_hidden_dense_layer_classifier'),
                             embedding_dim=hyper.get('embedding_dim'),
                             dropout_factor=hyper.get('drop_out'),
                             activation=activation
                             )
        model.summary()
        model_name = hyper.long_name()
        print(model_name)

        compile_model(model, optimizer="adam", learning_rate=hyper.get('learning_rate'), loss="categorical_crossentropy",
                      metrics=["accuracy"])
        history = train_model_lazy(model, early_stop_monitor=None, verbose=2, batch_size=hyper.get('batch_size'),
                                   epochs=10,
                                   x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                   log_file_name=LOG_DIR + model_name, csv_file_name= CSV_DIR + model_name)

def main_vary_one():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_7M)
    DATA_PCT = 0.01
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)

    for value in [1]:
        n_rnn_layers = 6
        n_class_layers = 1
        n_neurons_lstm_out = 256
        n_neurons_hidden_dense_layer_classifier = 512
        embedding_dim = 32
        learning_rate = 0.001
        batch_size = 256
        drop_out = 0
        activation = 'elu'

        hyper = HyperParams("RNN", x_train.shape[0],
                            n_rnn_layers=n_rnn_layers,
                            n_class_layers=n_class_layers,
                            n_neurons_lstm_out=n_neurons_lstm_out,
                            n_neurons_hidden_dense_layer_classifier=n_neurons_hidden_dense_layer_classifier,
                            embedding_dim=embedding_dim,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            drop_out=drop_out,
                            activation=activation,
                            )
        model = create_model(max_char_per_line=MAX_CHARS_PER_LINE,
                             vocabulary_size=VOCABULARY_SIZE,
                             number_of_classes=NUMBER_OF_CLASSES,
                             n_rnn_layers=hyper.get('n_rnn_layers'),
                             n_class_layers=hyper.get('n_class_layers'),
                             n_neurons_lstm_out=hyper.get('n_neurons_lstm_out'),
                             n_neurons_hidden_dense_layer_classifier=hyper.get('n_neurons_hidden_dense_layer_classifier'),
                             embedding_dim=hyper.get('embedding_dim'),
                             dropout_factor=hyper.get('drop_out'),
                             activation=activation
                             )
        model.summary()
        model_name = hyper.long_name()
        print(model_name)

        compile_model(model, optimizer="adam", learning_rate=hyper.get('learning_rate'), loss="categorical_crossentropy",
                      metrics=["accuracy"])
        history = train_model_lazy(model, early_stop_monitor=None, verbose=2, batch_size=hyper.get('batch_size'),
                                   epochs=10,
                                   x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                   log_file_name=LOG_DIR + model_name, csv_file_name= CSV_DIR + model_name)

def main_fine_tune():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_400M)
    DATA_PCT = 0.001
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)

, 
    for activation in ['relu', 'sigmoid']:
        for drop_out in [0, 0.2]:
            for n_rnn_layers in [6, 7]:
                for lstm in [False, True]:
                    n_class_layers = 1
                    n_neurons_lstm_out = 256
                    n_neurons_hidden_dense_layer_classifier = 512
                    embedding_dim = 32
                    learning_rate = 0.001
                    batch_size = 256

                    hyper = HyperParams("RNN", x_train.shape[0],
                                        n_rnn_layers=n_rnn_layers,
                                        n_class_layers=n_class_layers,
                                        n_neurons_lstm_out=n_neurons_lstm_out,
                                        n_neurons_hidden_dense_layer_classifier=n_neurons_hidden_dense_layer_classifier,
                                        embedding_dim=embedding_dim,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size,
                                        drop_out=drop_out,
                                        activation=activation,
                                        lstm=lstm,
                                        )
                    model = create_model(max_char_per_line=MAX_CHARS_PER_LINE,
                                         vocabulary_size=VOCABULARY_SIZE,
                                         number_of_classes=NUMBER_OF_CLASSES,
                                         n_rnn_layers=hyper.get('n_rnn_layers'),
                                         n_class_layers=hyper.get('n_class_layers'),
                                         n_neurons_lstm_out=hyper.get('n_neurons_lstm_out'),
                                         n_neurons_hidden_dense_layer_classifier=hyper.get('n_neurons_hidden_dense_layer_classifier'),
                                         embedding_dim=hyper.get('embedding_dim'),
                                         dropout_factor=hyper.get('drop_out'),
                                         activation=activation,
                                         lstm=lstm,
                                         )
                    model.summary()
                    model_name = hyper.long_name()
                    print(model_name)

                    compile_model(model, optimizer="adam", learning_rate=hyper.get('learning_rate'), loss="categorical_crossentropy",
                                  metrics=["accuracy"])
                    history = train_model_lazy(model, early_stop_monitor='val_loss', patience=10,
                                               verbose=2, batch_size=hyper.get('batch_size'),
                                               epochs=10,
                                               x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                               log_file_name=LOG_DIR + model_name, csv_file_name= CSV_DIR + model_name,
                                               model_file_name=MODELS_DIR + model_name, patience_lr=3)
                    evaluate_model(model, x_val, y_val)


def main_last_model():
    (x_train, y_train), (x_val, y_val) = load_data(PICKLE_FILE_NAMES_400M)
    DATA_PCT = 1
    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)
    
    
    #  El siguiente código hace que el acc de test no pase del 4% que es el valor obtenido por aleatoriedad
    #from sklearn.model_selection import train_test_split
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    n_class_layers = 2  #1
    n_neurons_lstm_out = 256 #256 
    n_neurons_hidden_dense_layer_classifier = 512
    embedding_dim = 32
    learning_rate = 0.0001 # 0.001 bajamos el learing rate porque se dispara el loss y no converge (puede durar mucho el entrenamiento)
    batch_size = 2048 #3072 # 256 lo subo a 3072 para que entrene más rápido
    activation = 'relu' # 'sigmoid'
    drop_out = 0.2 # Los mejores resultados los da para 0.2 pero se eterniza en el entrenamiento (con tantas instancias probablemente no necesite regularización)
    n_rnn_layers = 6  # 6 
    lstm = True, 

    hyper = HyperParams("RNN", x_train.shape[0],
                                        n_rnn_layers=n_rnn_layers,
                                        n_class_layers=n_class_layers,
                                        n_neurons_lstm_out=n_neurons_lstm_out,
                                        n_neurons_hidden_dense_layer_classifier=n_neurons_hidden_dense_layer_classifier,
                                        embedding_dim=embedding_dim,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size,
                                        drop_out=drop_out,
                                        activation=activation,
                                        lstm=lstm,
                                        )
    model = create_model(max_char_per_line=MAX_CHARS_PER_LINE,
                                         vocabulary_size=VOCABULARY_SIZE,
                                         number_of_classes=NUMBER_OF_CLASSES,
                                         n_rnn_layers=hyper.get('n_rnn_layers'),
                                         n_class_layers=hyper.get('n_class_layers'),
                                         n_neurons_lstm_out=hyper.get('n_neurons_lstm_out'),
                                         n_neurons_hidden_dense_layer_classifier=hyper.get('n_neurons_hidden_dense_layer_classifier'),
                                         embedding_dim=hyper.get('embedding_dim'),
                                         dropout_factor=hyper.get('drop_out'),
                                         activation=activation,
                                         lstm=lstm,
                                         )
    model.summary()
    model_name = hyper.long_name()
    print(model_name)

    compile_model(model, optimizer="adam", learning_rate=hyper.get('learning_rate'), loss="categorical_crossentropy",
                                  metrics=["accuracy"])
    history = train_model_lazy(model, early_stop_monitor='val_loss', patience=10,
                                               verbose=2, batch_size=hyper.get('batch_size'),
                                               epochs=12,
                                               x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                               log_file_name=LOG_DIR + model_name, csv_file_name= CSV_DIR + model_name,
                                               model_file_name=MODELS_DIR + model_name, patience_lr=3)
    evaluate_model(model, x_val, y_val)


if __name__ == "__main__":
    print("GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    disable_keras_warning_messages()
    #main_test()
    #main_train()  # val_acc=0.936308 con dos capas bidireccionales (batch=2048, neurons_lstm_out=128, neurons_classifier=512, embedding=32, class_layers=1, dropout=0)
                   # val_acc=0.942522 con tres capas bidireccionales
                   # val_acc=0.945825 con cuatro capas bidireccionales
                   # val_acc=0.948721 con cinco capas bidireccionales
                   # val_acc=0.949882 con seis capas bidireccionales
                   # val_acc=0.XXXXXX con siete capas bidireccionales (se quedó sin memoria para batchsize=2048)
    #main_vary_one()
    #main_fine_tune()
    , ,    # mejor modelo con acc=0.974 y loss de caida convergente a 0.09486
, ,    # n_class_layers = 1
, ,    # n_neurons_lstm_out = 256
, ,    # n_neurons_hidden_dense_layer_classifier = 512
, ,    # embedding_dim = 32
, ,    # learning_rate = 0.001
, ,    # batch_size = 256
, ,    # activation = 'sigmoid'
, ,    # drop_out = 0.2
, ,    # n_rnn_layers = 6
, ,    # lstm = True
    main_last_model()



