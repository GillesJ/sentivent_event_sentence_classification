#!/usr/bin/env python3
'''
multilabel_classification.py
sentivent_event_sentence_classification 
12/11/19
Copyright (c) Gilles Jacobs. All rights reserved.  
'''

from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
from pathlib import Path
import settings
from ast import literal_eval
import json

from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Load full dataset
dataset_fp = Path(settings.DATA_PROCESSED_DIRP) / "dataset_event_type.tsv"
dataset_df = pd.read_csv(dataset_fp, sep="\t", converters={"labels": literal_eval})

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
# TODO dev_df = dataset_df[dataset_df["dataset"] == "silver"]
# TODO train_df, eval_df = split_train_eval(dev_df)
train_df = dataset_df[dataset_df["dataset"] == "silver"]
test_df = dataset_df[dataset_df["dataset"] == "gold"]

num_labels = len(train_df["labels"][0])

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel(settings.MODEL_SETTINGS["model_type"],
                                      settings.MODEL_SETTINGS["model_name"],
                                      num_labels=num_labels,
                                      args=settings.MODEL_SETTINGS["train_args"])
print(train_df.head())

model_dirp = Path(settings.MODEL_DIRP) / f"{timestamp}-{settings.MODEL_SETTINGS['model_name']}/"

# Train the model
model.train_model(train_df, output_dir=model_dirp)

# write model settings
with open(model_dirp / "model_settings.json", "wt") as ms_out:
    json.dump(settings.MODEL_SETTINGS, ms_out)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)

test_df["y_pred"] = model_outputs.tolist()
test_df.to_csv(model_dirp / "test_predictions.tsv", sep="\t", index=False)

with open(model_dirp / "result.json", "wt") as result_out:
    json.dump(result, result_out)
print(result)
print(model_outputs)