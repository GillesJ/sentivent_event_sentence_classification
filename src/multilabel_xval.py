#!/usr/bin/env python3
'''
multilabel_xval
sentivent_event_sentence_classification 
12/18/19
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import numpy as np
from pathlib import Path
import settings
from ast import literal_eval
import json
import operator
from functools import reduce
from sklearn.model_selection import GroupKFold

from datetime import datetime

pd.options.mode.chained_assignment = None
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Load full dataset
dataset_fp = Path(settings.DATA_PROCESSED_DIRP) / "dataset_event_type.tsv"
dataset_df = pd.read_csv(dataset_fp, sep="\t", converters={"labels": literal_eval})

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
dev_df = dataset_df[dataset_df["dataset"] == "silver"]
test_df = dataset_df[dataset_df["dataset"] == "gold"]

num_labels = len(dev_df["labels"][0])

# Create a MultiLabelClassificationModel
print(f"Cross-validating across {settings.N_FOLDS} folds with model:\n{settings.MODEL_SETTINGS}")

model = MultiLabelClassificationModel(settings.MODEL_SETTINGS["model_type"],
                                      settings.MODEL_SETTINGS["model_name"],
                                      num_labels=num_labels,
                                      args=settings.MODEL_SETTINGS["train_args"])

model_dirp = Path(settings.MODEL_DIRP) / f"{timestamp}-{settings.MODEL_SETTINGS['model_name']}/"

# Make KFolds
group_kfold = GroupKFold(n_splits=settings.N_FOLDS)
groups = dev_df["document_id"].to_numpy()
X = dev_df["text"].to_numpy()
y = dev_df["labels"].to_numpy()

# train folds and collect results
results_df = pd.DataFrame(columns=["fold", "score", "predictions_fp"])
for i, (train_idc, eval_idc) in enumerate(group_kfold.split(X, y, groups)):
    print(f"Fold {i}: {train_idc.shape[0]} train inst. and {eval_idc.shape[0]} eval inst.")
    train_df = dev_df.iloc[train_idc]
    eval_df = dev_df.iloc[eval_idc]

    print(train_df.head())

    fold_dirp = model_dirp / f"{i}_fold"

    # Train the model
    model.train_model(train_df, output_dir=fold_dirp)
    # Evaluate the model on holdout test
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(f"Fold {i}: {result}")

    # collect and write predictions in fold dir
    eval_df["y_pred"] = model_outputs.tolist()
    predictions_fp = fold_dirp / "predictions.tsv"
    eval_df.to_csv(predictions_fp, sep="\t", index=False)

    with open(fold_dirp / "result.json", "wt") as result_out:
        json.dump(result, result_out)

    # collect fold results
    results_df = results_df.append({
        "fold": i,
        "score": result,
        "predictions_fp": predictions_fp,
    }, ignore_index=True)

# average fold results
results_df = results_df.append({
    "fold": "all_avg",
    "score": {key: np.mean([d.get(key) for d in results_df["score"].tolist()]) for key in
                  reduce(operator.or_, (d.keys() for d in results_df["score"].tolist()))},
    "predictions_fp": None,
    },
    ignore_index=True)
results_df = results_df.set_index("fold")
print(f"Crossvalidation score: {results_df.loc['all_avg', 'score']}")

# write crossval results
results_fp = model_dirp / "xval_holdout_results.tsv"
results_df.to_csv(results_fp, sep="\t")

# Retrain on full dev-set
print("Re-training on full dev set")
model.train_model(train_df, output_dir=model_dirp)
# Evaluate the model on holdout test
result, model_outputs, wrong_predictions = model.eval_model(test_df)
print(f"Test score: {result}")

holdout_predictions_fp = model_dirp / "holdouttest_predictions.tsv"
results_df = results_df.append({
    "fold": "holdouttest",
    "score": result,
    "predictions_fp": holdout_predictions_fp,
    },
    ignore_index=True)
results_df.to_csv(results_fp, sep="\t")

# write holdouttestset_df
test_df["y_pred"] = model_outputs.tolist()
test_df.to_csv(holdout_predictions_fp, sep="\t", index=False)

# write model settings
with open(model_dirp / "model_settings.json", "wt") as ms_out:
    json.dump(settings.MODEL_SETTINGS, ms_out)

print(f"Crossvalidation and holdout testing finished. Results in {model_dirp}")