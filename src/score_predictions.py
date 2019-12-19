#!/usr/bin/env python3
'''
score_predictions
sentivent_event_sentence_classification 
12/12/19
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import pandas as pd
from pprint import pprint
from ast import literal_eval
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import settings
import json
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, hamming_loss, label_ranking_loss, multilabel_confusion_matrix, ndcg_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score

model_dir = Path("/home/gilles/sentivent_event_sentence_classification/models/")

model_dirn = "2019-12-18_14-16-40-roberta-large/" # 4 epochs
model_dirn = "2019-12-18_16-22-09-roberta-large" # BEST YET 8 epochs
# model_dirn = "2019-12-18_16-59-12-roberta-large" # 16 epochs
model_dirp = model_dir / model_dirn

pred_fp = model_dirp / "test_predictions.tsv"
out_fp = model_dirp / "test_predictions_proc.tsv"
with open(Path(settings.DATA_PROCESSED_DIRP) / "type_classes_multilabelbinarizer.json", "rt") as classes_in:
    classes = json.load(classes_in)

testset_df = pd.read_csv(pred_fp, sep="\t", converters={"labels": literal_eval, "y_pred": literal_eval})

testset_df["y_pred_norm"] = np.array(testset_df["y_pred"].apply(lambda x: np.around(x).astype(int)))

# Type task: One-hot decode type labels
mlb = MultiLabelBinarizer()
mlb.fit([classes])
print(mlb.classes_)
testset_df["labels_pred"] = mlb.inverse_transform(np.array(testset_df["y_pred_norm"].to_list()))

testset_df.to_csv(out_fp, sep="\t", index=False)

y_true = np.array(testset_df["labels"].to_list())
y_pred = np.array(testset_df['y_pred'].to_list())
y_pred_norm = np.array(testset_df['y_pred_norm'].to_list())
# collect scores

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
                scores[f"precision{avg_suffix}"], scores[f"recall{avg_suffix}"], scores[f"f1{avg_suffix}"], _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
            except:
                scores[f"precision{avg_suffix}"], scores[f"recall{avg_suffix}"], scores[f"f1{avg_suffix}"] = None, None, None
            try:
                scores[f"roc_auc{avg_suffix}"] = roc_auc_score(y_true, y_pred, average=avg)
            except:
                scores[f"roc_auc{avg_suffix}"] = None
        else:
            try:
                p, r, f, _ = precision_recall_fscore_support(y_true, y_pred)
                scores[f"precision"], scores[f"recall"], scores[f"f1"] = (dict(zip(labels, list(sc))) for sc in (p, r, f))
            except:
                scores[f"precision"], scores[f"recall"], scores[f"f1"] = None, None, None
            try:
                scores["roc_auc"] = roc_auc_score(y_true, y_pred)
            except:
                scores["roc_auc"] = None

    return scores

scores = {}
scores["y_proba"] = get_score(y_true, y_pred, labels=classes)
scores["y_norm"] = get_score(y_true, y_pred_norm, labels=classes)

print("Probability prediction scores")
pprint(scores["y_proba"])
print("Discrete prediction scores")
pprint(scores["y_norm"])

cm = multilabel_confusion_matrix(y_true, y_pred_norm) # actually computes [[TN FN], [FP TP]] per class
records = []
for conf, label in zip(cm, classes):
    record = (label,
              {
                  "tn": conf[0, 0],
                  "tp": conf[1, 1],
                  "fn": conf[1, 0],
                  "fp": conf[0, 1],
              })
    pprint(record)
# TODO NTH get this in DF if prettier presentation is needed

# Get label counts in train and test
# Load full dataset
dataset_fp = Path(settings.DATA_PROCESSED_DIRP) / "dataset_event_type.tsv"
dataset_df = pd.read_csv(dataset_fp, sep="\t", converters={"labels": literal_eval})

train_labels = np.array(dataset_df[dataset_df["dataset"] == "silver"]["labels"].to_list())
test_labels = np.array(dataset_df[dataset_df["dataset"] == "gold"]["labels"].to_list())

train_label_sum = np.sum(train_labels, axis=0).astype(int)
test_label_sum = np.sum(test_labels, axis=0).astype(int)
train_label_pct = np.divide(np.multiply(train_label_sum, 100), np.sum(train_label_sum))
test_label_pct = np.divide(np.multiply(test_label_sum, 100), np.sum(test_label_sum))

label_df = pd.DataFrame(
    {
        "label_cnt_train": train_label_sum,
        "label_pct_train": train_label_pct,
        "label_cnt_test_true": test_label_sum,
        "label_pct_test_true": test_label_pct,
        "label_cnt_test_pred": np.sum(y_pred_norm, axis=0),
        "test_train_pct_delta": train_label_pct - test_label_pct,
        "f1": scores["y_norm"]["f1"],
        "recall": scores["y_norm"]["recall"],
        "precision": scores["y_norm"]["precision"]
    }, index=classes)

label_df.update(label_df.select_dtypes(include=np.number).applymap('{:,g}'.format))

label_df = label_df.astype({"label_cnt_train": int, "label_cnt_test_true": int})
print(label_df.round(2).to_html())

label_df