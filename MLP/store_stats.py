#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module appends statistics obtained during training, along with the total parameter count of the model,
to a given CSV file. The statistics are retrieved from a PrintAndSaveStats callback object and are
formatted into a row, which is then appended to the CSV file."""

import parameters
from callbacks import PrintAndSaveStats


def append_stats_after_train(stats_callback: PrintAndSaveStats, csv_name: str, separator: str, param_count: int) -> None:
    row: str = parameters.get_param_row(separator) + f"{param_count:,}" + separator
    row += separator.join([f"{stat:,}" for stat in stats_callback.get_stats()])
    file1 = open(csv_name, "a")
    file1.write(row + "\n")
    file1.close()
