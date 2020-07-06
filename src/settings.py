#!/usr/bin/env python3
"""
settings.py
sentivent_event_sentence_classification 
12/10/19
Copyright (c) Gilles Jacobs. All rights reserved.  
"""

RANDOM_STATE = 42
DATA_RAW_DIRP = "../data/raw/"
DATA_XMI_EXPORT_DIRN = "XMI-SENTiVENT-event-english-1.0-clean_2019-12-11_1246"
DATA_INTERIM_DIRP = "../data/interim/"
DATA_PROCESSED_DIRP = "../data/processed/"

MODEL_DIRP = (
    "/home/gilles/shares/lt3_sentivent/sentivent_event_sentence_classification/models/"
)
# MODEL_DIRP = "../models" # filled quickly

N_FOLDS = 10

MODEL_SETTINGS = {
    "model_type": "xlm",
    "model_name": "xlm-mlm-xnli15-1024",
    "train_args": {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 12,
        "n_gpu": 1,
    },
}


# N_FOLDS = 2 # for testing albert-base-v2
# MODEL_SETTINGS = {
#     "model_type": "albert",
#     "model_name": "albert-base-v2",
#     "train_args": {
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 1,
#         "n_gpu": 1,
#     },
# }

# MODEL_SETTINGS = {
#     "model_type": "xlnet",
#     "model_name": "xlnet-large-cased",
#     "train_args": {
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 32,
#         "n_gpu": 1,
#         "save_steps": 16384,
#         "save_model_every_epoch": False,
#         "fp16": False,
#     },
# }

# MODEL_SETTINGS = {
#     "model_type": "xlnet",
#     "model_name": "xlnet-base-cased",
#     "train_args": {
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 4,
#         "n_gpu": 1,
#         "save_steps": 16384,
#         "save_model_every_epoch": False,
#         "fp16": False,
#     },
# }

# MODEL_SETTINGS = {
#     "model_type": "roberta",
#     "model_name": "roberta-large-openai-detector",
#     "train_args": {
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 8,
#         "n_gpu": 1,
#         "save_steps": 16384,
#         "save_model_every_epoch": False,
#         "fp16": False,
#     },
# }

# MODEL_SETTINGS = {
#     "model_type": "albert",
#     "model_name": "albert-xxlarge-v2",
#     "train_args": {
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 12,
#         "n_gpu": 1,
#     },
# }
#
