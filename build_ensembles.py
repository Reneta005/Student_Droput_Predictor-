"""Train and cross-validate stage-specific soft-voting ensembles.

Members are chosen from the completed individual-model CV results: logistic
regression for recall, XGBoost for ranking/recall, and random forest where it
adds precision/accuracy. Soft voting uses equal weights, avoiding tuned blend
weights on the same validation folds.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from compare_datasets import (
    ORIG_S1, ORIG_S2, ORIG_CATS, OULAD_S1, OULAD_S2, OULAD_CATS,
    make_data, make_preprocessor, build_model, metric_row,
)


BASE = Path(__file__).parent
RESULTS = BASE / "results" / "dataset_comparison"
OUT_MODELS = RESULTS / "models"
ENSEMBLE_NAME = "CV-selected soft vote"
RECIPES = {
    ("Original", "Stage 1"): (ORIG_S1, ORIG_CATS, ["Logistic regression", "XGBoost"]),
    ("Original", "Stage 2"): (ORIG_S2, ORIG_CATS, ["Logistic regression", "XGBoost", "Random forest"]),
    ("OULAD", "Stage 1"): (OULAD_S1, OULAD_CATS, ["Logistic regression", "XGBoost"]),
    ("OULAD", "Stage 2"): (OULAD_S2, OULAD_CATS, ["XGBoost", "Random forest"]),
}


def make_pipeline(columns, categorical, member, y_train, seed):
    return Pipeline([
        ("preprocess", make_preprocessor(columns, categorical)),
        ("model", build_model(member, y_train, seed=seed)),
    ])


def voting(members, columns, categorical, y_train, seed):
    estimators = [(re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"),
                   make_pipeline(columns, categorical, name, y_train, seed))
                  for name in members]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)


def main():
    datasets = make_data(BASE / "cleaned_data.csv", BASE / "oulad_student_level.csv")
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    rows, predictions, artifacts, recipe_manifest = [], [], {}, {}

    for dataset_name, data in datasets.items():
        target = data.target.to_numpy()
        splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                    if dataset_name == "OULAD" else StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        splits = list(splitter.split(np.zeros(len(data)), target,
                                    groups=data.group.to_numpy() if dataset_name == "OULAD" else None))
        for stage in ("Stage 1", "Stage 2"):
            columns, categorical, members = RECIPES[(dataset_name, stage)]
            oof_probability = np.zeros(len(data), dtype=float)
            oof_fold = np.zeros(len(data), dtype=int)
            for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
                model = voting(members, columns, categorical, target[train_idx], seed=42 + fold)
                model.fit(data.iloc[train_idx][columns], target[train_idx])
                oof_probability[valid_idx] = model.predict_proba(data.iloc[valid_idx][columns])[:, 1]
                oof_fold[valid_idx] = fold
                print(f"{dataset_name} {stage}: fold {fold}/5 complete ({'+'.join(members)})", flush=True)
            oof_prediction = (oof_probability >= 0.5).astype(int)
            row = metric_row(dataset_name, stage, ENSEMBLE_NAME, target, oof_prediction,
                             oof_probability, "5-fold OOF")
            row["ensemble_members"] = " + ".join(members)
            rows.append(row)
            predictions.extend({"dataset": dataset_name, "stage": stage, "row_index": int(i),
                                "fold": int(oof_fold[i]), "target": int(target[i]),
                                "probability": float(oof_probability[i]),
                                "prediction_at_0_50": int(oof_prediction[i])}
                               for i in range(len(data)))

            # Final refit on all available rows; CV metrics above remain the evaluation.
            final = voting(members, columns, categorical, target, seed=42)
            final.fit(data[columns], target)
            filename = f"{dataset_name.lower()}_{stage.lower().replace(' ', '_')}_soft_vote.joblib"
            dump(final, OUT_MODELS / filename)
            artifacts[f"{dataset_name} {stage}"] = str(Path("models") / filename)
            recipe_manifest[f"{dataset_name} {stage}"] = {
                "members": members,
                "features": columns,
                "target": "Dropout" if dataset_name == "Original" else "Withdrawn from a module-presentation",
                "artifact": str(Path("models") / filename),
            }
            print(f"Saved final refit: {OUT_MODELS / filename}", flush=True)

    ensemble_metrics = pd.DataFrame(rows)
    ensemble_metrics.to_csv(RESULTS / "selected_ensemble_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(RESULTS / "selected_ensemble_oof_predictions.csv", index=False)
    (RESULTS / "selected_ensemble_setup.json").write_text(json.dumps({
        "method": "equal-weight soft voting of predict_proba values",
        "threshold": 0.5,
        "members_by_dataset_stage": recipe_manifest,
        "cross_validation": "stratified 5-fold; grouped by student for OULAD",
        "training_note": "Final serialized estimators are refit on all rows; use CV metrics for estimated generalization.",
    }, indent=2), encoding="utf-8")

    cv_path = RESULTS / "within_dataset_cv.csv"
    previous = pd.read_csv(cv_path)
    previous = previous.loc[previous.model != ENSEMBLE_NAME]
    combined = pd.concat([previous, ensemble_metrics], ignore_index=True, sort=False)
    combined.to_csv(cv_path, index=False)

    transfer_path = RESULTS / "cross_dataset_shared_features.csv"
    transfer = pd.read_csv(transfer_path)
    md = ["# Model comparison", "",
          "Within-dataset results use pooled out-of-fold predictions from stratified 5-fold CV (grouped by student for OULAD). Models use class weighting; MLP uses balanced sample weights computed on training folds only. The CV-selected ensembles use equal soft voting and a 0.50 threshold.", "",
          "## Within-dataset CV", "", combined.to_markdown(index=False, floatfmt=".3f"), "",
          "## Cross-dataset transfer", "",
          "These tests use only harmonized gender and approximate age band. OULAD Withdrawn is withdrawal from a module-presentation, not necessarily university dropout. Its outcome differs from the original institution's Dropout vs Enrolled/Graduate grouping, so transfer metrics are exploratory and not a direct validation of university dropout prediction.", "",
          transfer.to_markdown(index=False, floatfmt=".3f"), "",
          "## Interpretation", "",
          "Accuracy is reported, but recall is the priority. Stage 2 generally improves within-dataset discrimination. Ensembles should be retained only if their selected operational metric improves over the strongest single model. Cross-dataset AUC is near chance with the limited shared features; there is no evidence of reliable transfer between these sources.", ""]
    (RESULTS / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    print("\nSelected ensemble CV results:\n", ensemble_metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
