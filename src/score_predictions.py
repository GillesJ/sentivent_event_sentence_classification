#!/usr/bin/env python3
"""
Computes several scores given a model directory containing training-test run subdirs with a `testset_with_predictions.tsv` file.
Output:
- In each run subdir: a processed `testset_with_predictions.tsv` file with human-readable predictions per instance for error analysis.
- In each run subdir: scores with metrics (json)
- In root model dir: summary results.tsv file with holdout scores + averaged crossvalidation scores.
- In root model dir: class_scores.tsv overview of scores by class/type in holdout test and crossvalidation.

score_predictions
sentivent_event_sentence_classification 
12/12/19
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import operator
from functools import reduce
import pandas as pd
from pprint import pprint
from ast import literal_eval
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import settings
import json
from sklearn.metrics import (
    coverage_error,
    label_ranking_average_precision_score,
    hamming_loss,
    label_ranking_loss,
    multilabel_confusion_matrix,
    ndcg_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)


def flatten_embedded_dict(dictionary):
    return {
        (outerKey, innerKey): values
        for outerKey, innerDict in dictionary.items()
        for innerKey, values in innerDict.items()
    }


def get_score(y_true, y_pred, labels=None):
    scores = {}
    scores["lrap"] = label_ranking_average_precision_score(y_true, y_pred)
    scores["lrloss"] = label_ranking_loss(y_true, y_pred)
    scores["ndcg_score"] = ndcg_score(y_true, y_pred)
    scores["coverage_error"] = coverage_error(y_true, y_pred)
    try:
        scores["hamming_loss"] = hamming_loss(y_true, y_pred)
    except:
        scores["hamming_loss"] = None
    try:
        scores["subset_accuracy"] = accuracy_score(y_true, y_pred)
    except:
        scores["subset_accuracy"] = None

    for avg in [None, "micro", "macro", "weighted", "samples"]:
        if avg:
            avg_suffix = f"_{avg}"
            try:
                (
                    scores[f"precision{avg_suffix}"],
                    scores[f"recall{avg_suffix}"],
                    scores[f"f1{avg_suffix}"],
                    _,
                ) = precision_recall_fscore_support(y_true, y_pred, average=avg)
            except:
                (
                    scores[f"precision{avg_suffix}"],
                    scores[f"recall{avg_suffix}"],
                    scores[f"f1{avg_suffix}"],
                ) = (None, None, None)
            try:
                scores[f"roc_auc{avg_suffix}"] = roc_auc_score(
                    y_true, y_pred, average=avg
                )
            except:
                scores[f"roc_auc{avg_suffix}"] = None
        else:
            try:
                p, r, f, _ = precision_recall_fscore_support(y_true, y_pred)
                scores[f"precision"], scores[f"recall"], scores[f"f1"] = (
                    dict(zip(labels, list(sc))) for sc in (p, r, f)
                )
            except:
                scores[f"precision"], scores[f"recall"], scores[f"f1"] = (
                    None,
                    None,
                    None,
                )
            try:
                scores["roc_auc"] = roc_auc_score(y_true, y_pred)
            except:
                scores["roc_auc"] = None

    return scores


# 1. Set model path and load labels and predictions
model_dir = Path("/home/gilles/sentivent_event_sentence_classification/models/")

# model_dirn = "2019-12-18_14-16-40-roberta-large/" # OLD-RUN 4 epochs
# model_dirn = "2019-12-18_16-22-09-roberta-large" # OLD-RUN BEST YET 8 epochs
# model_dirn = "2019-12-26_22-39-44-roberta-large" # OLD-RUN
# model_dirn = "2019-12-18_16-59-12-roberta-large" # 16 epochs
# model_dirn = "2020-01-06_10-38-14-roberta-large"  # 2fold test run with mistake
model_dirn = "2020-01-06_14-41-02-roberta-large"  # 2fold test run
model_dirn = "2020-01-06_14-41-59-roberta-large"  # 10fold run 8 epochs
model_dirn = "2020-01-07_16-17-03-roberta-large" # 10fold run 6 epochs (BEST Loss, slightly worse micro F1 than 8 epochs)
model_dirp = model_dir / model_dirn

# load classes json as made by parse_to_processed.py
with open(
    Path(settings.DATA_PROCESSED_DIRP) / "type_classes_multilabelbinarizer.json", "rt"
) as classes_in:
    classes = json.load(classes_in)

# Load result_df
result_df = pd.read_csv(model_dirp / "results.tsv", sep="\t", index_col="run_name")
result_df = result_df.dropna()  # drop the average axis

all_scores = {}

# Iterate over run dirs and process testset results + collect fold scores
for run_name, row in result_df.iterrows():

    testset_fp = Path(row["run_dirp"]) / "testset_with_predictions.tsv"

    print(f"-------\nProcessing {run_name} for {model_dirn}\n-------")

    # Set output path
    testset_proc_fp = (
        testset_fp.parent / f"{testset_fp.stem}_processed{''.join(testset_fp.suffixes)}"
    )

    # Load preds
    testset_df = pd.read_csv(
        testset_fp,
        sep="\t",
        converters={"labels": literal_eval, "y_pred": literal_eval},
    )

    # Binarize multilabel preds
    testset_df["y_pred_bin"] = np.array(
        testset_df["y_pred"].apply(lambda x: np.around(x).astype(int))
    )

    # Get labels from binarized y: One-hot decode type labels.
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    testset_df["labels_pred"] = mlb.inverse_transform(
        np.array(testset_df["y_pred_bin"].to_list())
    )
    testset_df.to_csv(testset_proc_fp, sep="\t", index=False)  # write processed file

    # Get the predictions
    y_true = np.array(testset_df["labels"].to_list())
    y_pred = np.array(testset_df["y_pred"].to_list())
    y_pred_bin = np.array(testset_df["y_pred_bin"].to_list())

    # Collect scores
    scores = {}
    scores["y_proba"] = get_score(y_true, y_pred, labels=classes)
    scores["y_bin"] = get_score(y_true, y_pred_bin, labels=classes)

    all_scores[run_name] = scores

    print("Probability prediction scores")
    pprint(scores["y_proba"])
    print("Discrete prediction scores")
    pprint(scores["y_bin"])

    cm = multilabel_confusion_matrix(
        y_true, y_pred_bin
    )  # actually computes [[TN FN], [FP TP]] per class
    records = []
    for conf, label in zip(cm, classes):
        record = {
            "label": label,
            "tn": conf[0, 0],
            "tp": conf[1, 1],
            "fn": conf[1, 0],
            "fp": conf[0, 1],
        }
        records.append(record)
    cm_df = pd.DataFrame(records)
    # TODO NTH get this in DF if prettier presentation is needed

    # Get label counts in train and test
    # Load full dataset for comparative stats
    trainset_fp = testset_fp.parent / "trainset.tsv"
    trainset_df = pd.read_csv(
        trainset_fp, sep="\t", converters={"labels": literal_eval}
    )

    train_labels = np.array(trainset_df["labels"].to_list())
    test_labels = np.array(testset_df["labels"].to_list())

    train_label_sum = np.sum(train_labels, axis=0).astype(int)
    test_label_sum = np.sum(test_labels, axis=0).astype(int)
    train_label_pct = np.divide(
        np.multiply(train_label_sum, 100), np.sum(train_label_sum)
    )
    test_label_pct = np.divide(np.multiply(test_label_sum, 100), np.sum(test_label_sum))

    label_df = pd.DataFrame(
        {
            "label_cnt_train": train_label_sum,
            "label_pct_train": train_label_pct,
            "label_cnt_test_true": test_label_sum,
            "label_pct_test_true": test_label_pct,
            "label_cnt_test_pred": np.sum(y_pred_bin, axis=0),
            "test_train_pct_delta": train_label_pct - test_label_pct,
            "f1": scores["y_bin"]["f1"],
            "recall": scores["y_bin"]["recall"],
            "precision": scores["y_bin"]["precision"],
        },
        index=classes,
    )

    label_df.update(
        label_df.select_dtypes(include=np.number).applymap("{:,g}".format)
    )  # format human readable

    # write all scores as json, scores by label as tsv, and confusion matrix
    label_df.to_csv(testset_fp.parent / "label_scores.tsv", sep="\t")
    cm_df.to_csv(
        testset_fp.parent / "confusion_matrix_multilabel.tsv", sep="\t", index=False
    )
    with open(testset_fp.parent / "scores.json", "wt") as scores_out:
        json.dump(scores, scores_out, indent=4, sort_keys=True)


def average_fold_df(fold_df):
    type_cols = ["precision", "recall", "f1"]
    fold_types_df = fold_df[type_cols]
    fold_no_types_df = fold_bin_df.drop(type_cols, axis=1)
    fold_avg_df = fold_no_types_df.mean(axis=0)
    type_avg_scores = []
    for c in type_cols:
        score_dicts = fold_types_df[c].to_list()
        avg = {
            key: np.mean([d.get(key) for d in score_dicts])
            for key in reduce(operator.or_, (d.keys() for d in score_dicts))
        }
        avg["metric"] = c
        type_avg_scores.append(avg)
    fold_types_avg_df = pd.DataFrame(type_avg_scores).set_index("metric").transpose()

    return fold_avg_df, fold_types_avg_df


# write code for walking a dict and averaging same keys
fold_scores = [all_scores[k] for k in all_scores if k != "holdout"]

fold_bin_df = pd.DataFrame([d["y_bin"] for d in fold_scores])
fold_proba_df = pd.DataFrame([d["y_proba"] for d in fold_scores])

fold_bin_avg_df, fold_bin_types_avg_df = average_fold_df(fold_bin_df)
fold_proba_avg_df = fold_proba_df.dropna(axis=1).mean()

types_metrics = ["precision", "recall", "f1"]
holdout_df = pd.DataFrame(all_scores["holdout"])
holdout_df = holdout_df.drop(types_metrics, axis=0)
holdout_types = {k: all_scores["holdout"]["y_bin"][k] for k in types_metrics}
holdout_types_df = pd.DataFrame(holdout_types)

summary_df = pd.concat(
    [holdout_df, fold_proba_avg_df, fold_bin_avg_df], axis=1, sort=True
)
summary_df.columns = ["Holdout_proba", "Holdout_bin", "Crossval_proba", "Crossval_bin"]
summary_df = summary_df[
    ["Crossval_bin", "Holdout_bin", "Crossval_proba", "Holdout_proba"]
]

summary_types_df = pd.concat(
    [fold_bin_types_avg_df, holdout_types_df],
    keys=["Crossvalidation", "Holdout"],
    axis=1,
    sort=True,
)

# write summaries
summary_df.update(summary_df.select_dtypes(include=np.number).applymap("{:,g}".format))
summary_df.to_csv(model_dirp / "score_summary.tsv", sep="\t")
summary_types_df.update(
    summary_types_df.select_dtypes(include=np.number).applymap("{:,g}".format)
)
summary_types_df.to_csv(model_dirp / "score_by_type_summary.tsv", sep="\t")
print(f"{model_dirn.upper()} holdout and crossvalidation scores summary:")
print(summary_df)
print(f"{model_dirn.upper()} holdout and crossvalidation scores by class label summary:")
print(summary_types_df)