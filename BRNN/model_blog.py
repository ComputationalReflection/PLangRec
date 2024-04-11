#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that models a transformer-based language classifier
"""

import os
import pickle
from typing import List

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential, Model
import keras



class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate) if dropout_rate > 0 else None
        self.dropout2 = Dropout(dropout_rate) if dropout_rate > 0 else None

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training) if self.dropout1 else attn_output
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training) if self.dropout2 else ffn_output
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


def create_model(n_attention_heads, n_neurons_hidden_dense_layer, max_char_per_line, vocabulary_size,
                 embedding_dim, number_of_classes: int, dropout_factor: float, n_neurons_hidden_classifier:int):
    inputs = Input(shape=(max_char_per_line,))
    embedding_layer = TokenAndPositionEmbedding(max_char_per_line, vocabulary_size, embedding_dim)
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
                      dropout_factor: float, n_neurons_hidden_classifier: int, n_layers_hidden_classifier: int):
    inputs = Input(shape=(max_words_per_review,))
    embedding_layer = TokenAndPositionEmbedding(max_words_per_review, vocabulary_size, embedding_dim)
    x = embedding_layer(inputs)
    for i in range(n_transformer_blocks):
        transformer_block = TransformerBlock(embedding_dim, n_attention_heads, n_neurons_hidden_dense_layer)
        x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    if dropout_factor > 0:
        x = Dropout(dropout_factor)(x)
    for i in range(n_layers_hidden_classifier):
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

