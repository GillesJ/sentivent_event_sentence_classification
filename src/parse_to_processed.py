#!/usr/bin/env python3
'''
parse_to_processed.py

Script for one experimental run for .

sentivent_event_sentence_classification 
12/10/19
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import sys
sys.path.append("/home/gilles/repos/")

from sentivent_webannoparser import parse_project
from pathlib import Path
import settings
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import json

# parse the corpus
corpus = parse_project.parse_project(Path(settings.DATA_RAW_DIRP) / settings.DATA_XMI_EXPORT_DIRN)

# select useful info for sentence level event type clf to Dataframe
data = []
for k, v in corpus.items():
    for doc in v:
        for i, sentence in enumerate(doc.sentences):
            type_labels = [ev.event_type for ev in sentence.events]
            subtype_labels = [f"{ev.event_type}.{ev.event_subtype}" for ev in sentence.events]
            instance = {
                "document_id": doc.document_id,
                "document_title": doc.title,
                "sentence_idx": i,
                "text": str(sentence),
                "types_event": type_labels,
                "subtypes_event": subtype_labels,
                "types_event_unq": list(set(type_labels)),
                "subtypes_event_unq": list(set(subtype_labels)),
                "token_cnt": len(sentence.tokens),
                "event_cnt": len(sentence.events),
                "dataset": k,
            }
            print(instance)
            data.append(instance)

dataset_df = pd.DataFrame().from_records(data)
dataset_df["sentence_idx"] = dataset_df["sentence_idx"].astype(int)
dataset_df["token_cnt"] = dataset_df["token_cnt"].astype(int)
dataset_df["event_cnt"] = dataset_df["event_cnt"].astype(int)

# Write interim dataframe
dataset_df.to_csv(Path(settings.DATA_INTERIM_DIRP) / "dataset.tsv", sep="\t", index=False)
# TODO Add company metadata to each record: company_id, company_name

# Type task: One-hot encode type labels
mlb = MultiLabelBinarizer()
labels_list = np.array(dataset_df["types_event_unq"].tolist())
labels = mlb.fit_transform(labels_list)
print(f"{len(mlb.classes_)} classes: {mlb.classes_}")
dataset_df["labels"] = labels.tolist()

dataset_df.to_csv(Path(settings.DATA_PROCESSED_DIRP) / "dataset_event_type.tsv", sep="\t", index=False)
with open(Path(settings.DATA_PROCESSED_DIRP) / "type_classes_multilabelbinarizer.json", "wt") as classes_out:
    json.dump(mlb.classes_.tolist(), classes_out)

# Subtype task: One-hot encode subtype labels
mlb = MultiLabelBinarizer()
labels_list = np.array(dataset_df["subtypes_event_unq"].tolist())
labels = mlb.fit_transform(labels_list)
print(f"{len(mlb.classes_)} classes: {mlb.classes_}")
dataset_df["labels"] = labels.tolist()

dataset_df.to_csv(Path(settings.DATA_PROCESSED_DIRP) / "dataset_event_subtype.tsv", sep="\t", index=False)
with open(Path(settings.DATA_PROCESSED_DIRP) / "subtype_classes_multilabelbinarizer.json", "wt") as classes_out:
    json.dump(mlb.classes_.tolist(), classes_out)