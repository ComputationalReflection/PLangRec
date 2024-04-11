#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class modeling the hyperparameters of neural networks
"""

class HyperParams:

    def __init__(self, model_name: str, n_individuals: int, batch_size: int,
                 n_attention_heads: int = None, n_trans_blocks: int = None,
                 n_rnn_layers: int = None,
                 n_neurons_hidden_dense_layer_encoder: int = None, drop_out: int = None,
                 n_neurons_hidden_dense_layer_classifier: int = None, n_class_layers: int = None,
                 learning_rate: int = None, n_neurons_lstm_out: int = None,
                 embedding_dim: int = None, activation: str = None, lstm: bool = None):
        self.properties = dict()
        self.model_name = model_name
        self.properties['n_individuals'] = n_individuals
        self.properties['batch_size'] = batch_size
        if n_attention_heads is not None:
            self.properties['n_attention_heads'] = n_attention_heads
        if n_trans_blocks is not None:
            self.properties['n_trans_blocks'] = n_trans_blocks
        if n_rnn_layers is not None:
            self.properties['n_rnn_layers'] = n_rnn_layers
        if n_neurons_hidden_dense_layer_encoder is not None:
            self.properties['n_neurons_hidden_dense_layer_encoder'] = n_neurons_hidden_dense_layer_encoder
        if drop_out is not None:
            self.properties['drop_out'] = drop_out
        if n_neurons_hidden_dense_layer_classifier is not None:
            self.properties['n_neurons_hidden_dense_layer_classifier'] = n_neurons_hidden_dense_layer_classifier
        if n_class_layers is not None:
            self.properties['n_class_layers'] = n_class_layers
        if learning_rate is not None:
            self.properties['learning_rate'] = learning_rate
        if n_neurons_lstm_out is not None:
            self.properties['n_neurons_lstm_out'] = n_neurons_lstm_out
        if embedding_dim is not None:
            self.properties['embedding_dim'] = embedding_dim
        if activation is not None:
            self.properties['activation'] = activation
        if lstm is not None:
            self.properties['lstm'] = lstm

    def get(self, name: str) -> any:
        return self.properties[name]

    def set(self, name: str, value: any):
        self.properties[name] = value

    def long_name(self):
        result = f"{self.model_name}-{self.properties['n_individuals']}-"
        for key, value in self.properties.items():
            if value is not None and key != 'n_individuals':
                result += f"{key}_{value}-"
        return result
