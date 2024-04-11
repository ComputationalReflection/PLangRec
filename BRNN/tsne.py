#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that visualizes in 2D the dispersion of the input individuals, and compare it
to the distribution of the same individuals after performing the forward pass of the ANN.
It graphically shows how the ANN is capable of classifying the individuals.
"""

import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from configuration import PICKLE_FILE_NAMES_400M

MODEL_FILE_NAME = "models/RNN-432180483-batch_size_2048-n_rnn_layers_8-drop_out_0-n_neurons_hidden_dense_layer_classifier_512-n_class_layers_2-learning_rate_0.0001-n_neurons_lstm_out_256-embedding_dim_32-activation_relu-lstm_True-"
LANGUAGE_LABELS = ["Assembly", "C", "C++", "C#", "CSS", "Go", "HTML", "Java", "JavaScript", "Kotlin",
                   "Matlab", "Perl", "PHP", "Python", "R", "Ruby", "Scala", "SQL", "Swift", "TypeScript",
                   "Unix Shell"]
NUMBER_INSTANCES_TO_PLOT = 1_000

def visualize_tsne(input_data, y_labels, text_of_labels, title):
    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(input_data)
    # Create a scatter plot with different colors for each class
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, 21))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', '+', 'x', 'X', 'D', 'd',
               'o', 's', '^', 'v', '<', '>', 'p', '*', '+', 'x', 'X', 'D', 'd']
    for i, label in enumerate(unique_labels):
        print(f"Processing label {label} ({text_of_labels[label]})...")
        class_indices = y_labels == label
        plt.scatter(reduced_data[class_indices, 0], reduced_data[class_indices, 1],
                    #c=[colors[i]],
                    color=colors[i],
                    marker=markers[i], label=text_of_labels[label])
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.ion()
    plt.show()

def load_dataset(x_file_name: str, y_file_name: str):
    print("Loading X for validation...")
    with open(x_file_name, 'rb') as handle:
        x_test = pickle.load(handle)
    print("Loading Y validation...")
    with open(y_file_name, 'rb') as handle:
        y_test = pickle.load(handle)
    # Convert from one-hot to integer values
    y_test = np.argmax(y_test, axis=1)
    print(x_test.shape)
    print(y_test.shape)
    return x_test, y_test


def get_feature_from_model(x_input, model, feature_index: int):
    extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    all_features = extractor(x_input)
    feature = all_features[feature_index].numpy()
    return feature


def shuffle_dataset(x_data, y_data):
    assert len(x_data) == len(y_data)
    indices = np.random.permutation(len(x_data))
    return x_data[indices], y_data[indices]

def main():
    # Load dataset
    x_test, y_test = load_dataset(PICKLE_FILE_NAMES_400M['x_test'], PICKLE_FILE_NAMES_400M['y_test'])
    x_test, y_test = shuffle_dataset(x_test, y_test)
    visualize_tsne(x_test[:NUMBER_INSTANCES_TO_PLOT], y_test[:NUMBER_INSTANCES_TO_PLOT], LANGUAGE_LABELS, "Original dataset")
    # Load model
    print(f"Loading the model from {MODEL_FILE_NAME} ...")
    model = keras.models.load_model(MODEL_FILE_NAME)
    # Get last feature (layer) from model
    last_feature = get_feature_from_model(x_test[:NUMBER_INSTANCES_TO_PLOT], model, -2)
    # Visualize the last feature from model
    visualize_tsne(last_feature, y_test[:NUMBER_INSTANCES_TO_PLOT], LANGUAGE_LABELS, "Last layer of the BRNN")
    plt.show(block=True)


if __name__ == "__main__":
    main()
