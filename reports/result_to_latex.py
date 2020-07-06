#!/usr/bin/env python3
'''
result_to_latex.py
sentivent_event_sentence_classification 
2/13/20
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import matplotlib.pyplot as plt
import pandas as pd
from src import settings
from pathlib import Path
import numpy as np

plt.close('all')
plt.figure();
model_rundir = Path(settings.MODEL_DIRP) / "roberta-large_epochs-16"

# load all scores
all_scores_df = pd.read_csv(model_rundir / "score_summary.tsv", sep="\t").rename(columns={"Unnamed: 0": "Event type"}).set_index("Event type")
typescore_df = pd.read_csv(model_rundir / "score_by_type_summary.tsv", sep="\t").rename(columns={"Unnamed: 0": "Event type"}).set_index("Event type")

def plot_type_score(typescore_df, columns):
    typesc_df = typescore_df[columns]
    typesc_df.columns = typesc_df.loc["metric"]
    typesc_df = typesc_df.drop(index="metric")
    for c in typesc_df.columns:
        typesc_df[c] = typesc_df[c].astype(np.float64)

    typesc_df = typesc_df.sort_values(by=["f1"])
    ax = typesc_df.plot.bar(width=0.75)
    ax.legend(["Precision", "Recall", "F1-score"])
    ax.set_xticklabels(list(typesc_df.index), rotation=45, ha="right")
    for i, p in enumerate(sorted(ax.patches, key=lambda x: x.xy)):
        if (i+1) % 3 == 0:
            ax.annotate(f"{np.round(p.get_height(),decimals=2)}".replace("0.","."), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    # load type results and show only holdout
    fn = f"type-score-{columns[0].lower()}"
    plt.savefig(model_rundir / (fn + ".svg"), bbox_inches="tight")
    plt.savefig(model_rundir / (fn + ".png"), bbox_inches="tight")

    plt.show()
    plt.close('all')

plot_type_score(typescore_df, ["Holdout", "Holdout.1", "Holdout.2"])
plot_type_score(typescore_df, ["Crossvalidation", "Crossvalidation.1", "Crossvalidation.2"])
pass
#