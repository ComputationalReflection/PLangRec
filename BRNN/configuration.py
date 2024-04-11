#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file with global constants
"""

import math

# Data
DATA_PCT = 0.001
MODELS_DIR = './models/'
LOG_DIR = './logs/'
CSV_DIR = './csvs/'
PICKLE_FILE_NAMES_7M = {
    "x_train": "pkl/7_000_000_train.pkl",
    "y_train": "pkl/7_000_000_train_labels.pkl",
    "x_val": "pkl/7_000_000_valid.pkl",
    "y_val": "pkl/7_000_000_valid_labels.pkl",
    "model": "7_000_000"
}
PICKLE_FILE_NAMES_70M = {
    "x_train": "pkl/70_000_000_train.pkl",
    "y_train": "pkl/70_000_000_train_labels.pkl",
    "x_val": "pkl/70_000_000_valid.pkl",
    "y_val": "pkl/70_000_000_valid_labels.pkl",
    "model": "70_000_000"
}
PICKLE_FILE_NAMES_400M = {
    "x_train": "pkl/432_180_483_train.pkl",
    "y_train": "pkl/432_180_483_train_labels.pkl",
    "x_val": "pkl/1_000_020_valid.pkl",
    "y_val": "pkl/1_000_020_valid_labels.pkl",
    "x_test": "pkl/1_000_020_test.pkl",
    "y_test": "pkl/1_000_020_test_labels.pkl",
    "model": "400_000_000"
}
PICKLE_FILE_NAMES_700M = {
    "x_train": "pkl/700_000_000_train.pkl",
    "y_train": "pkl/700_000_000_train_labels.pkl",
    "x_val": "pkl/700_000_000_valid.pkl",
    "y_val": "pkl/700_000_000_valid_labels.pkl",
    "x_test": "pkl/1_000_000_test.pkl",
    "y_test": "pkl/1_000_000_test_labels.pkl",
    "model": "700_000_000"
}

# Input
VOCABULARY_SIZE = 97  # Size of the vocabulary
MAX_CHARS_PER_LINE = 40  # Max number of characters per line
NUMBER_OF_CLASSES = 21  # Number of languages (different targets)


# Model hyper-parameters
EMBEDDING_DIM = value if (value := math.ceil(math.log2(VOCABULARY_SIZE))) % 2 == 0 else value + 1
EMBEDDING_DIM *= 4

# Programming languages that can be classified
LANGUAGES = ["Assembly", "C", "C++", "C#", "CSS", "Go", "HTML", "Java", "JavaScript", "Kotlin",
                   "Matlab", "Perl", "PHP", "Python", "R", "Ruby", "Scala", "SQL", "Swift", "TypeScript",
                   "Unix Shell"]
