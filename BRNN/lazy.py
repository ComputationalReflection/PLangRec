#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic experiment that uses the lazypredict package to test some ML models
"""

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from configuration import PICKLE_FILE_NAMES_7M
from data import load_data, select_first_in_list
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from configuration import VOCABULARY_SIZE
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from time import time
from lazypredict.Supervised import LazyClassifier
import pandas as pd

def to_one_hot_one_dimension(values: np.ndarray, max_value: int) -> np.ndarray:
    onehot = np.zeros((values.size, max_value+1))
    onehot[np.arange(values.size), values] = 1
    flattened = onehot.flatten().astype(int)
    return flattened

def to_one_hot(values: np.ndarray, max_value: int = None) -> np.ndarray:
    max_value = max_value if max_value else values.max()
    result = np.apply_along_axis(lambda row: to_one_hot_one_dimension(row, max_value), axis=1, arr=values)
    return result

def _test_to_one_hot():
    values = np.array([
        np.array([0, 1, 2]),
        np.array([2, 3, 0]),
        np.array([1, 1, 4])
    ])
    flattened = to_one_hot(values)
    print(values)
    print(flattened)

def to_arg_max(values: np.ndarray) -> np.ndarray:
    result = np.apply_along_axis(lambda row: np.argmax(row), axis=1, arr=values)
    return result

def main():
    # Data processing
    DATA_PCT = 0.1
    file_names = PICKLE_FILE_NAMES_7M
    (x_train, y_train), (x_val, y_val) = load_data(file_names)

    x_train, y_train, x_val, y_val = select_first_in_list(x_train, DATA_PCT),  select_first_in_list(y_train, DATA_PCT), \
                                      select_first_in_list(x_val, DATA_PCT), select_first_in_list(y_val, DATA_PCT)
    print(f"Shapes of x_train={x_train.shape}, y_train={y_train.shape}.")
    print(f"Shapes of x_val={x_val.shape}, y_val={y_val.shape}.")


    print("Converting data...")
    x_train, x_val = to_one_hot(x_train), to_one_hot(x_val, max_value=x_train.max())
    y_train, y_val = to_arg_max(y_train), to_arg_max(y_val)
    print(f"Shapes of x_train={x_train.shape}, y_train={y_train.shape}.")
    print(f"Shapes of x_val={x_val.shape}, y_val={y_val.shape}.")

    print("Training all the classifiers...")
    pd.options.display.max_columns = 2000
    #best_classifiers = [LinearSVC, PassiveAggressiveClassifier, XGBClassifier, ExtraTreesClassifier,
    #                    RandomForestClassifier, RidgeClassifier, GaussianNB, SGDClassifier]
    best_classifiers = [XGBClassifier, ExtraTreesClassifier, RandomForestClassifier]
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, classifiers=best_classifiers)
    models, predictions = clf.fit(x_train, x_val, y_train, y_val)
    print(models)

if __name__ == "__main__":
    main()


