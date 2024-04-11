#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that models a transformer-based language classifier
"""

import pickle
from typing import List

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential, Model
import keras
from datetime import datetime

from configuration import MAX_WORDS_PER_REVIEW
from data import convert_statements_to_embedding_lists
import numpy as np

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def create_model(n_attention_heads, n_neurons_hidden_dense_layer, max_words_per_review, vocabulary_size,
                 embedding_dim, number_of_classes: int, dropout_factor: float, n_neurons_hidden_classifier:int):
    inputs = Input(shape=(max_words_per_review,))
    embedding_layer = TokenAndPositionEmbedding(max_words_per_review, vocabulary_size, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, n_attention_heads, n_neurons_hidden_dense_layer)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    x = Dense(n_neurons_hidden_classifier, activation="relu")(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    outputs = Dense(number_of_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_deep_model(n_attention_heads:int, n_transformer_blocks:int, n_neurons_hidden_dense_layer:int,
                      max_words_per_review:int, vocabulary_size:int, embedding_dim:int, number_of_classes: int,
                      dropout_factor: float, n_neurons_hidden_classifier:int):
    inputs = Input(shape=(max_words_per_review,))
    embedding_layer = TokenAndPositionEmbedding(max_words_per_review, vocabulary_size, embedding_dim)
    x = embedding_layer(inputs)
    for i in range(n_transformer_blocks):
        transformer_block = TransformerBlock(embedding_dim, n_attention_heads, n_neurons_hidden_dense_layer)
        x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    x = Dense(n_neurons_hidden_classifier, activation="relu")(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    outputs = Dense(number_of_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_model_double_pooling(n_attention_heads, n_neurons_hidden_dense_layer, max_words_per_review, vocabulary_size,
                                embedding_dim, number_of_classes: int, dropout_factor: float,
                                n_neurons_hidden_classifier:int):
    inputs = Input(shape=(max_words_per_review,))
    embedding_layer = TokenAndPositionEmbedding(max_words_per_review, vocabulary_size, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, n_attention_heads, n_neurons_hidden_dense_layer)
    x = transformer_block(x)
    pool1 = GlobalAveragePooling1D()(x)
    pool2 = GlobalMaxPooling1D()(x)
    concat = keras.layers.Concatenate()([pool1, pool2])
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(concat)
    x = Dense(n_neurons_hidden_classifier, activation="relu")(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    outputs = Dense(number_of_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_model_max_pooling(n_attention_heads, n_neurons_hidden_dense_layer, max_words_per_review, vocabulary_size,
                             embedding_dim, number_of_classes: int, dropout_factor: float,
                             n_neurons_hidden_classifier:int):
    inputs = Input(shape=(max_words_per_review,))
    embedding_layer = TokenAndPositionEmbedding(max_words_per_review, vocabulary_size, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, n_attention_heads, n_neurons_hidden_dense_layer)
    x = transformer_block(x)
    x = GlobalMaxPooling1D()(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    x = Dense(n_neurons_hidden_classifier, activation="relu")(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    outputs = Dense(number_of_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def compile_model(model, optimizer:str, loss:str, metrics:List[str]):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train_model(model, early_stop_monitor:str, patience: int, verbose:int, batch_size:int, epochs:int,
                x_train, y_train, x_val, y_val, patience_lr: int = 4):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=patience, verbose=verbose,
                                                           restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=patience_lr, min_lr=0.00001, verbose=verbose)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./models/',
        save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[early_stop_callback, model_checkpoint_callback])
    return history



def __generate_data(x, y, batch_size, epochs:int):
    for epoch in range(epochs):
        for n in range(x.shape[0]//batch_size):
            from_index, to_index = n*batch_size, (n+1)*batch_size
            yield np.array(x[from_index:to_index]), np.array(y[from_index:to_index])
        yield np.array(x[to_index:]), np.array(y[to_index:])


def train_model_lazy(model, early_stop_monitor:str, patience:int, verbose:int, batch_size:int, epochs:int,
                x_train, y_train, x_val, y_val, dir: str, patience_lr: int = 4):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=patience, verbose=verbose,
                                                           restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=patience_lr, min_lr=0.00001, verbose=verbose)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./models/',
        save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
    for epoch in range(1, epochs+1):
        model.fit(__generate_data(x_train, y_train, batch_size, epochs=1),
                        steps_per_epoch = x_train.shape[0]//batch_size+1,
                                  epochs=1, validation_data=(x_val, y_val),
                        callbacks=[early_stop_callback, model_checkpoint_callback])
        loss, accuracy = evaluate_model(model, x_val, y_val)
        save_model(dir, model, accuracy)

def evaluate_model(model, x_val, y_val):
    results = model.evaluate(x_val, y_val, verbose=2)
    for name, value in zip(model.metrics_names, results):
        print("%s: %f" % (name, value), end=", ")
    print()
    return results

def save_model(dir, model, accuracy):
    file_name = f"{dir}accuracy_{accuracy}_{str(datetime.now())}.pkl"
    file_name = file_name.replace(" ", "_").replace(":", "_")
    with open(file_name, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved with name: {file_name}.")

def infer_from_model(model, x_val, y_val, sentence_converter, sentences):
    # Inference with validation set (x_val)
    y_predicted = model.predict(x_val)
    print("Inference (val):")
    for i in range(5):
        print(f"\t#{i}, ground truth: {y_val[i]}, predicted: '{y_predicted[i]}', "
              f"sentence: {sentence_converter.embedding_list_to_sentence(x_val[i])}")
    # Inference with test set (sentences)
    print("\nInference (test):")
    x_test = convert_statements_to_embedding_lists(sentence_converter, sentences, MAX_WORDS_PER_REVIEW)
    y_predicted = model.predict(x_test)
    for i in range(len(x_test)):
        print(f"\t#{i}, predicted: '{y_predicted[i]}', "
              f"sentence: {sentences[i]}, "
              f"*** actual sentence: {sentence_converter.embedding_list_to_sentence(x_test[i])}")


