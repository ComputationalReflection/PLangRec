#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module handles the storage of the last trained model and its training history. It saves the model in
HDF5 format with the name provided, along with its training history as a JSON file. Additionally, it plots the
training history using Pandas and Matplotlib, and saves the plot as a PNG file with the same name as the model. """

import json
import pandas
import matplotlib.pyplot as pyplot


def get_last_model_path(model_name: str) -> str:
    return model_name + "_last.h5"


def store_last_model(model_name: str, model, history) -> None:
    model.save(get_last_model_path(model_name))
    if history is None:
        return
    print(history.history)

    with open(model_name + '.json', 'w') as f:
        json.dump(history.history, f)

    pandas.DataFrame(history.history).plot(figsize=(8, 5))
    pyplot.grid(True)
    pyplot.gca().set_ylim(0, 1)
    pyplot.savefig(model_name + ".png")
