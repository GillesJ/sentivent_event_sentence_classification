#!/usr/bin/env python3
'''
settings.py
sentivent_event_sentence_classification 
12/10/19
Copyright (c) Gilles Jacobs. All rights reserved.  
'''

DATA_RAW_DIRP = "../data/raw/" # TODO copy this to ../data/raw/
DATA_XMI_EXPORT_DIRN = "XMI-SENTiVENT-event-english-1.0-clean_2019-12-11_1246"
DATA_INTERIM_DIRP = "../data/interim/"
DATA_PROCESSED_DIRP = "../data/processed/"

MODEL_DIRP = "../models/"

MODEL_TYPE = "roberta"
MODEL_NAME = "roberta-large"

# MODEL_SETTINGS = {
#     "model_type": "roberta",
#     "model_name": "roberta-large",
#     "train_args": {
#         'reprocess_input_data': True,
#         'overwrite_output_dir': True,
#         'num_train_epochs': 8,
#         'n_gpu': 4,
#     }
# }

MODEL_SETTINGS = {
    "model_type": "albert",
    "model_name": "albert-xxlarge-v2",
    "train_args": {
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'num_train_epochs': 8,
        'n_gpu': 1,
        "evaluate_during_training": False, # TODO TEST THIS
    }
}

N_FOLDS = 10