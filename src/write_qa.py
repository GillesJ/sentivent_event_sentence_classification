#!/usr/bin/env python3
'''
Write predictions of holdout to file for qualitative analysis.
Highlight columns with prediction errors, drop unnecessary info.

write_qa.py in sentivent_event_sentence_classification
6/29/20 Copyright (c) Gilles Jacobs
'''
import pandas as pd
from ast import literal_eval
from collections import Counter
from pathlib import Path

dirp = Path("/home/gilles/repos/sentivent_event_sentence_classification/models/roberta-large_epochs-16/holdout/")
in_file = dirp / "testset_with_predictions_processed.tsv"
annotated_fp = dirp / "qualitative_error_analysis_annotated.csv"


df_anno = pd.read_csv(annotated_fp, sep="\t",
                      converters={
                          "types_event_unq": literal_eval,
                          "labels_pred": literal_eval,
                      })
# # add column for extra labels
# df_anno["new_pred"] =(df_anno["labels_pred"].apply(set) - df_anno["types_event_unq"].apply(set)).apply(len) >= 1
# df_anno.to_csv(dirp / "qualitative_error_analysis.csv", sep="\t", index=False)

total_len = len(df_anno)
error_len = len(df_anno[df_anno["error"] == True])

# this analysis has to happen before we drop Na
pct_new_plau_new = df_anno[df_anno["new_pred"] == True]["plausible new label"].value_counts(normalize=True, dropna=False)["y"]

cols_anno = ["lexical cue", "true label ambiguity", "idiomatic context", "ambiguous trigger", "example worthy"]
df_anno = df_anno.dropna(axis=0, subset=cols_anno, how="all")
pct_anno = round((len(df_anno) * 100) / error_len, 2)

df = pd.read_csv(in_file, sep="\t", converters={"types_event_unq": literal_eval, "labels_pred": literal_eval})

# count yes ("y") annotations
pct_trigger_ambi = df_anno["ambiguous trigger"].value_counts(normalize=True, dropna=False)["y"]
pct_label_ambi = df_anno["true label ambiguity"].value_counts(normalize=True, dropna=False)["y"]
pct_context_idio = df_anno["idiomatic context"].value_counts(normalize=True, dropna=False)["y"]


# count pct new plausible
pct_new_plau_all = df_anno["plausible new label"].value_counts(normalize=True, dropna=False)["y"]


# count weak/strong cues
s = 0
w = 0
for i, row in df_anno.iterrows():
    try:
        labels = literal_eval(row["missclf"])
        cue = row["lexical cue"]
        strong_cnt = cue.count("strong")
        weak_cnt = cue.count("weak")
        if strong_cnt + weak_cnt > 1:
            s += strong_cnt
            w += weak_cnt
        else:
            if strong_cnt == 1 and weak_cnt == 0:
                s += len(labels)
            if weak_cnt == 1 and strong_cnt == 0:
                w += len(labels)
    except AttributeError: # skip Nan where other annotation is made
        pass

weak_pct = round(100 * w / (w + s), 1)
strong_pct = round(100 * s / (w + s), 1)

print(f"Weak cues {weak_pct}% ({w}/{w+s})")
print(f"Strong cues {strong_pct}% ({s}/{w+s})")
cols_keep = ["document_id", "sentence_idx", "text", "types_event_unq", "labels_pred"]
df = df[cols_keep]

true = list(df["types_event_unq"].apply(set))
pred = list(df["labels_pred"].apply(set))
miss = [y_t ^ y_p for y_t, y_p in zip(true, pred)]
df["missclf"] = miss
df["error"] = df["missclf"] != set()

df.to_csv(dirp / "qualitative_error_analysis.csv", sep="\t")

miss_cnt = Counter(i for st in miss for i in st)

labels = list(miss_cnt.keys())
missclass_matrix = {}
for y_t, y_p in zip(true, pred):
    y_p = set(["none"]) if not y_p else y_p
    y_wrong = y_t ^ y_p
    for x in y_wrong:

        if x in y_t:
            missclass_matrix.setdefault(x, Counter()).update(list(y_wrong - y_t))
pass