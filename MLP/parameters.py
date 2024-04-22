#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing all the parameters supported by and used to run both dataset preprocessing and model training
scripts. It stores and sets default values for not specified parameters and infers the best value for parameters in
case they depend on another one. Also, it also provides names and paths to load and store both models and datasets,
depending on the configuration specified by the user. It allows the user to execute generate_dataset.py and train.py
scripts with different configurations by supporting the introduction of the parameters keys via command-line along
with its value in the form: python train.py  batch_size=1024 epochs=20 embedding_size=56 n_denses=8... It also
supports the introduction of several configurations by specifying lists instead of values for the
different parameters: python train.py batch_size=[512,1024] epochs=20 n_denses=[6,8,10]...
To train the models, use the TRAIN_PATH and VALID_PATH args to specify train set pickle and the validation set pickle.
To generate datasets, use the RAW_CSV_TRAIN_PATH and RAW_CSV_VALID_PATH args to specify the source csv files.
To generate just a test/validation set, the argument ONLY_VALID must be set to True.
The name of the arguments to be introduced via command-line is not case-sensitive.
"""

import sys
import os
from copy import copy

MIN_POS = 32
MAX_POS = 127
DS_FOLDER = "datasets"
RESULTS_FOLDER = "results"
LOOKUP_FOLDER = "lookups"
GLOBAL_CSV = "train_stats.csv"
TENSORBOARD_DIR = "C:\\tensorboard_logs"

DEF_MAX_LEN = 40
DEF_VOCAB_SIZE = 97
SELU_ACTIVATION = "selu"
LEAKY_RELU_ACTIVATION = "leaky_relu"
LECUN_INITIALIZER = "lecun_normal"
HE_INITIALIZER = "he_normal"

# KEYS
ACTIVATION_FUNC = "ACTIVATION"
BALANCED = "BALANCED"
BATCH_SIZE = "BATCH_SIZE"
DENSES_WIDTH = "DENSES_WIDTH"
DROPOUT = "DROPOUT"
EMBEDDING_SIZE = "EMBEDDING_SIZE"
EPOCHS = "EPOCHS"
EXP_NAME = "EXP_NAME"
FLATTEN_DOWN = "FLATTEN_DOWN"
LAZY_LOAD = "LAZY_LOAD"
LOOKUP_FILE = "FEATURES_LOOKUP"
L_RATE = "L_RATE"
MAX_LEN = "MAX_LEN"
N_DENSES = "N_DENSES"
N_LABELS = "N_LABELS"
N_SAMPLES = "N_SAMPLES"
N_VALID = "N_VALID"
ONE_HOT = "ONE_HOT"
LOOKUP_OOV = "OOV_CHARS"
PATIENCE = "PATIENCE"
PICKLE_ONLY = "PICKLE_ONLY"
PICKLE_SIZE = "PICKLE_SIZE"
SHUFFLE = "SHUFFLE"
START = "START"
RAW_CSV_TRAIN_PATH = "RAW_CSV_TRAIN_PATH"
RAW_CSV_VALID_PATH = "RAW_CSV_VALID_PATH"
DS_TRAIN_PATH = "TRAIN_PATH"
DS_VALID_PATH = "VALID_PATH"
VOCAB_SIZE = "VOCAB_SIZE"
WEIGHT_INITIALIZER = "WEIGHT_INITIALIZER"
GENERATE_ONLY_VALID = "ONLY_VALID"
INT_LABEL = "INT_LABEL"
DEVICE_NAME = "DEVICE_NAME"


def init_args() -> dict:
    arg_map = {}

    arg_map[ACTIVATION_FUNC] = SELU_ACTIVATION
    arg_map[BALANCED] = True
    arg_map[BATCH_SIZE] = 2048
    arg_map[DENSES_WIDTH] = int(DEF_MAX_LEN * DEF_VOCAB_SIZE / 10)
    arg_map[DROPOUT] = 0.0
    arg_map[EMBEDDING_SIZE] = None
    arg_map[EPOCHS] = 500
    arg_map[FLATTEN_DOWN] = True
    arg_map[LAZY_LOAD] = False
    arg_map[LOOKUP_FILE] = "features_lookup.tf"
    arg_map[MAX_LEN] = 40
    arg_map[N_DENSES] = 3
    arg_map[N_LABELS] = 21
    arg_map[N_SAMPLES] = 700_000
    arg_map[N_VALID] = None
    arg_map[ONE_HOT] = False
    arg_map[LOOKUP_OOV] = 0
    arg_map[GENERATE_ONLY_VALID] = False
    arg_map[PATIENCE] = 10
    arg_map[PICKLE_ONLY] = False
    arg_map[PICKLE_SIZE] = None
    arg_map[SHUFFLE] = True
    arg_map[START] = 1
    arg_map[RAW_CSV_TRAIN_PATH] = 'raw_train\\*.csv'
    arg_map[RAW_CSV_VALID_PATH] = 'raw_validation\\*.csv' #'raw_test\\*.csv'
    arg_map[DS_TRAIN_PATH] = 'dataset\\432_180_483_train'
    arg_map[DS_VALID_PATH] = 'dataset\\1_000_020_valid' #'dataset\\1_000_020_test'
    arg_map[EXP_NAME] = 'logs'
    arg_map[VOCAB_SIZE] = None
    arg_map[WEIGHT_INITIALIZER] = None
    arg_map[L_RATE] = 0.001
    arg_map[INT_LABEL] = False
    arg_map[DEVICE_NAME] = None

    return arg_map


ARG_MAP = init_args()


def set_arg_map(new_params: dict) -> None:
    for key, value in new_params.items():
        ARG_MAP[key] = value


def get_device_name():
    if not ARG_MAP[DEVICE_NAME] is None:
        return ARG_MAP[DEVICE_NAME]
    GPU_ENV_INDEX = 8
    ACTIVE_ENV_MARKER = "*"
    CONDA_COMMAND = "conda env list"
    env_split = os.popen(CONDA_COMMAND).read().split("\n")
    if len(env_split) < GPU_ENV_INDEX:
        return "CPU"
    return "GPU" if ACTIVE_ENV_MARKER in env_split[GPU_ENV_INDEX] else "CPU"


DEVICE_NAME = get_device_name()


def check_and_create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def create_dirs() -> None:
    check_and_create_dir(RESULTS_FOLDER)
    check_and_create_dir(DS_FOLDER)
    check_and_create_dir(LOOKUP_FOLDER)


def _parse(value: str, parser):
    if value.startswith("[") and value.endswith("]"):
        return [parser(elem) for elem in value[1:-1].split(",")]
    return parser(value)


def parse_args(arg_list: list, arg_separator: str = "=") -> dict:
    result = init_args()
    for arg_pair in arg_list:
        split = arg_pair.split(arg_separator)
        param = split[0].upper()

        if param == RAW_CSV_TRAIN_PATH or param == RAW_CSV_VALID_PATH or param == LOOKUP_FILE or param == ACTIVATION_FUNC \
                or param == WEIGHT_INITIALIZER or param == EXP_NAME or param == DS_TRAIN_PATH or param == DS_VALID_PATH\
                or param == DEVICE_NAME:
            result[param] = _parse(split[1], str)
        else:
            if split[1].upper() == "NONE":
                result[param] = None
            else:
                result[param] = _parse(split[1], eval)

    return result


def _add_to_all(attr_name: str, attr_value, configs: list) -> None:
    for config in configs:
        config[attr_name] = attr_value


def _expand(attr_name, attr_values: list, configs: list) -> list:
    result = []

    for value in attr_values:
        for config in configs:
            c = copy(config)
            c[attr_name] = value
            result.append(c)

    return result


def _cross_args(args: dict) -> list:
    result = [{}]

    for name, val in args.items():
        if isinstance(val, list):
            result = _expand(name, val, result)
        else:
            _add_to_all(name, val, result)

    return result


def process_args(arg_separator: str = "=") -> list:
    arg_list: list = []
    if len(sys.argv) > 0:
        arg_list = sys.argv[1:]

    configs = _cross_args(parse_args(arg_list))
    for config in configs:
        _set_derived_args(config)
        for param in config.keys():
            if isinstance(config[param], int):
                print(f"Param {param} : {config[param]:_} ({type(config[param])})")
            else:
                print(f"Param {param} : {config[param]} ({type(config[param])})")

    create_dirs()

    return configs


def _set_derived_args(args: dict) -> None:
    if args[N_VALID] == None:
        args[N_VALID] = 1_000_000 if args[N_SAMPLES] >= 10_000_000 else int(args[N_SAMPLES] / 10)

    if args[EMBEDDING_SIZE] != None:
        if args[ONE_HOT]:
            raise ValueError(f"One hot cannot be TRUE if embedding size is different"
                             f" than None ({args[EMBEDDING_SIZE]})")
    elif not args[ONE_HOT]:
        args[EMBEDDING_SIZE] = 40

    if args[WEIGHT_INITIALIZER] == None:
        if args[ACTIVATION_FUNC] == SELU_ACTIVATION:
            args[WEIGHT_INITIALIZER] = LECUN_INITIALIZER
        else:
            args[WEIGHT_INITIALIZER] = HE_INITIALIZER

    if not args[BALANCED] and args[LAZY_LOAD]:
        raise ValueError(f"Weighted classes is not implemented for lazy load yet."
                         f" BALANCED AND LAZY_LOAD CANNOT BE BOTH TRUE")

    if args[LOOKUP_OOV] == None:
        if args[VOCAB_SIZE] == None:
            args[VOCAB_SIZE] = MAX_POS - MIN_POS + 2  # No char = 0 and OOV_CHAR=1

    if args[PICKLE_SIZE] == None:
        args[PICKLE_SIZE] = args[N_SAMPLES]

    if args[START] != None:
        args[START] -= 1


def get_common_suffixes() -> str:
    suffix: str = ""

    if ARG_MAP[LOOKUP_OOV] != None:
        suffix += f"_{ARG_MAP[VOCAB_SIZE]}vocab"
        if ARG_MAP[LOOKUP_OOV] != 0:
            suffix += f"_{ARG_MAP[LOOKUP_OOV]}oov"
    else:
        suffix += "_def_vocab"

    if ARG_MAP[BALANCED]:
        suffix += "_balanced"
    else:
        suffix += "_stratified"

    suffix += f"_{ARG_MAP[MAX_LEN]}max_len"

    return suffix


def get_preprocess_suffixes() -> (str, str):
    suffix: str = ""
    if ARG_MAP[ONE_HOT]:
        suffix += "_one_hot"
    if ARG_MAP[INT_LABEL]:
        suffix += "_int_label"
    suffix += get_common_suffixes()

    return suffix + f"_{ARG_MAP[N_SAMPLES]:_}", suffix + f"_{ARG_MAP[N_VALID]:_}"


def get_pickle_paths() -> (str, str):
    train: str
    valid: str
    train, valid = get_preprocess_suffixes()
    pickle_folder: str = DS_FOLDER + "/" + train + "_pickle/"

    check_and_create_dir(pickle_folder)

    train_name: str = pickle_folder + f"{ARG_MAP[N_SAMPLES]:_}_train"
    val_name: str = pickle_folder + f"{ARG_MAP[N_VALID]:_}_valid"
    return train_name, val_name


def get_csv_paths() -> (str, str):
    train: str
    valid: str
    train, valid = get_preprocess_suffixes()

    train_name: str = DS_FOLDER + "/train" + train + ".csv"
    val_name: str = DS_FOLDER + "/valid" + valid + ".csv"
    return train_name, val_name


def get_model_name() -> str:
    suffix: str = DEVICE_NAME

    suffix += f"_{ARG_MAP[DROPOUT]}dropout"

    if ARG_MAP[ONE_HOT]:
        suffix += "_one_hot"
    else:
        suffix += f"_{ARG_MAP[EMBEDDING_SIZE]}embedding"

    suffix += get_common_suffixes()

    suffix += f"_{ARG_MAP[L_RATE]}_lr"
    suffix += f"_{ARG_MAP[ACTIVATION_FUNC]}_activation"

    suffix += f"_{ARG_MAP[N_DENSES]}_denses"

    suffix += f"_{ARG_MAP[DENSES_WIDTH]}_denseswidth"

    suffix += f"_{ARG_MAP[BATCH_SIZE]:_}batch"
    train_name: str = RESULTS_FOLDER + "/" + suffix + f"_{ARG_MAP[N_SAMPLES]:_}"
    return train_name


def get_lookup_file():
    maxlen = f"_{ARG_MAP[MAX_LEN]}maxlen" if ARG_MAP[MAX_LEN] != 40 else ""
    return LOOKUP_FOLDER + f"/{ARG_MAP[N_SAMPLES]}" + maxlen + f"_{ARG_MAP[VOCAB_SIZE]}vocab_" + ARG_MAP[LOOKUP_FILE]


def get_lookup_pickle():
    maxlen = f"_{ARG_MAP[MAX_LEN]}maxlen" if ARG_MAP[MAX_LEN] != 40 else ""
    return LOOKUP_FOLDER + f"/{ARG_MAP[N_SAMPLES]}" + maxlen + f"_ds_lookup"


def get_param_row(separator: str) -> str:
    return f"{DEVICE_NAME}{separator}{ARG_MAP[LAZY_LOAD]}{separator}{ARG_MAP[N_SAMPLES]:,}{separator}" \
           f"{ARG_MAP[MAX_LEN]}{separator}{ARG_MAP[BALANCED]}{separator}" \
           f"{ARG_MAP[FLATTEN_DOWN]}{separator}{ARG_MAP[N_DENSES]}{separator}{ARG_MAP[DENSES_WIDTH]}" \
           f"{separator}{ARG_MAP[BATCH_SIZE]:,}{separator}{ARG_MAP[EPOCHS]}{separator}{ARG_MAP[PATIENCE]}{separator}" \
           f"{ARG_MAP[EMBEDDING_SIZE]}{separator}{ARG_MAP[LOOKUP_OOV]}{separator}{ARG_MAP[VOCAB_SIZE]}{separator}" \
           f"{ARG_MAP[ACTIVATION_FUNC]}{separator}{ARG_MAP[WEIGHT_INITIALIZER]}{separator}"


def get_tensorboard_path() -> str:
    return TENSORBOARD_DIR + "\\" + ARG_MAP[EXP_NAME] + "\\" + get_model_name()


def get_hp_param_map() -> dict:
    hp_params = ARG_MAP.copy()
    hp_params.pop(FLATTEN_DOWN)
    hp_params.pop(LAZY_LOAD)
    hp_params.pop(LOOKUP_FILE)
    hp_params.pop(N_VALID)
    hp_params.pop(GENERATE_ONLY_VALID)
    hp_params.pop(PICKLE_ONLY)
    hp_params.pop(PICKLE_SIZE)
    hp_params.pop(SHUFFLE)
    hp_params.pop(START)
    hp_params.pop(RAW_CSV_TRAIN_PATH)
    hp_params.pop(RAW_CSV_VALID_PATH)
    hp_params.pop(DS_TRAIN_PATH)
    hp_params.pop(DS_VALID_PATH)
    hp_params.pop(EXP_NAME)
    hp_params.pop(LOOKUP_OOV)
    hp_params.pop(INT_LABEL)
    hp_params.pop(DEVICE_NAME)

    return hp_params
