#!/usr/bin/env python3
"""
rank_models.py
sentivent_event_sentence_classification 
1/27/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import score_predictions, settings
from pathlib import Path
import json
import pandas as pd

if __name__ == "__main__":
    FROM_SCRATCH = False

    # 1. Set model paths
    all_models_dirps_unclean = list(
        p for p in Path(settings.MODEL_DIRP).iterdir() if p.is_dir()
    )

    # Remove incomplete runs that did not train all folds + holdout
    all_models_dirps = []
    for p in all_models_dirps_unclean:
        subdirecall_cnt = sum(1 for x in p.iterdir() if x.is_dir())
        if subdirecall_cnt >= settings.N_FOLDS + 1:
            all_models_dirps.append(p)

    # 2. load classes
    with open(
        Path(settings.DATA_PROCESSED_DIRP) / "type_classes_multilabelbinarizer.json",
        "rt",
    ) as classes_in:
        classes = json.load(classes_in)

    # 3.Score preds and write summary
    run_names = [str(m.name) for m in all_models_dirps]
    all_scores_df = pd.DataFrame({"run": run_names, "path": all_models_dirps})
    all_scores_df = all_scores_df.set_index(["run"])
    for run, row in all_scores_df.iterrows():
        model_dirp = row["path"]
        print(
            f"______________________________\n{run.upper()}\n______________________________"
        )
        if not (model_dirp / "score_summary.tsv").exists() or FROM_SCRATCH:
            print(f"Scoring run summary from scratch.")
            summary_df, _ = score_predictions.make_score_summaries(model_dirp, classes)
        else:
            print(f"Loading run score summary.")
            summary_df = pd.read_csv(model_dirp / "score_summary.tsv", sep="\t").set_index(["Unnamed: 0"])

        # get scores
        metrics = ["precision_macro", "recall_macro", "f1_macro", "precision_micro", "recall_micro", "f1_micro", "roc_auc", "lrap", "ndcg_score", "subset_accuracy"]
        for metric in metrics:
            xval_score = float(summary_df.loc[metric, "Crossval_bin"])
            xval_std = summary_df.loc[metric, "Crossval_bin_std"]
            ho_score = summary_df.loc[metric, "Holdout_bin"]

            all_scores_df.loc[run, metric + "_crossval"] = xval_score
            all_scores_df.loc[run, metric + "_std"] = xval_std
            all_scores_df.loc[run, metric + "_holdout"] = ho_score
            all_scores_df.loc[run, metric + "_crossval_latex"] = f"{xval_score:.3f}  $\pm${xval_std:.3f}"

    # sort rows
    all_scores_df = all_scores_df.sort_values(["f1_macro_crossval"])
    # write ranking
    all_scores_df.to_csv(Path(settings.MODEL_DIRP) / "model_ranking.tsv", sep="\t", float_format="%.3f")

    #make nice for latex
    #filter bad exp:
    latex_df = all_scores_df.reset_index()
    latex_df = latex_df[~latex_df.run.str.startswith("2")]
    metrics = ["precision_macro", "recall_macro", "f1_macro"]
    display = []
    for d in ["_crossval_latex", "_holdout"]:
        for m in metrics:
            display.append(m + d)



    def make_model_name(mn):
        if not "DUMMY" in mn:
            name, pars = mn.split("_")
            pars = ' '.join(reversed(pars.split('-')))
            return f"{name.title()} ({pars})"
        else:
            dummy_strat = mn.replace("DUMMY-", "")
            return f"{dummy_strat} baseline"

    latex_df["Model"] = latex_df["run"].map(make_model_name)

    print(latex_df[["Model"]+display].to_latex(index=False, float_format='%.3f', escape=False, caption="Macro-averaged precision, recall, F1-score for sentence-level event type detection in crossvalidation and on the holdout set.", label="tab:allscores"))
