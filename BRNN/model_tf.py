#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that models a transformer-based language classifier
"""

import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x



class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        layers = [tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)]
        if dropout_rate > 0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
        self.seq = tf.keras.Sequential(layers)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        # Add dropout.
        x = self.dropout(x) if self.dropout else x
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # Shape `(batch_size, seq_len, d_model)`.



class Classifier(tf.keras.layers.Layer):
    def __init__(self, *, encoder, n_neurons_ff: int, number_of_classes: int, dropout_rate: int = 0.1):
        super().__init__()
        self.encoder = encoder
        self.pooling1d = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.ff = tf.keras.layers.Dense(n_neurons_ff, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.softmax = tf.keras.layers.Dense(number_of_classes, activation="softmax")

    def call(self, x):
        x = self.encoder(x)
        x = self.pooling1d(x)
        x = self.dropout1(x) if self.dropout1 else x
        x = self.ff(x)
        x = self.dropout2(x) if self.dropout2 else x
        output = self.softmax(x)
        return output


class EncoderClassifierModel(tf.keras.Model):
    def __init__(self, *, n_transformer_blocks, embedding_dim, n_attention_heads,
                 n_neurons_hidden_dense_layer, vocabulary_size,
                 n_neurons_hidden_classifier: int, number_of_classes: int,
                 dropout_factor):
        super().__init__()
        encoder = Encoder(num_layers=n_transformer_blocks, d_model=embedding_dim,
                               num_heads=n_attention_heads, dff=n_neurons_hidden_dense_layer,
                               vocab_size=vocabulary_size, dropout_rate=dropout_factor)
        self.classifier = Classifier(encoder=encoder, n_neurons_ff=n_neurons_hidden_classifier,
                                     dropout_rate=dropout_factor, number_of_classes=number_of_classes)

    def call(self, x):
        output = self.classifier(x)
        return output

