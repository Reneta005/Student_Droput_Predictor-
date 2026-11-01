"""Five-fold, group-aware model comparison for the original dataset and OULAD.

Also evaluates cross-institution transfer on two intentionally narrow shared
features (gender and age band). Results retain each dataset's original outcome
definition; cross-dataset scores are exploratory, not perfectly equivalent.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ORIG_S1 = ["Admission grade", "Previous qualification (grade)", "Age at enrollment", "Course",
           "Daytime/evening attendance", "Scholarship holder", "Debtor",
           "Gender", "Education Special Needs"]
ORIG_S2 = ORIG_S1 + ["Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"]
ORIG_CATS = {"Course", "Daytime/evening attendance", "Scholarship holder",
             "Debtor", "Gender", "Education Special Needs"}
OULAD_S1 = ["gender", "region", "highest_education", "imd_band", "age_band",
            "num_of_prev_attempts", "studied_credits", "disability", "code_module", "code_presentation"]
OULAD_S2 = OULAD_S1 + ["early_assessments_submitted", "early_mean_assessment_score",
                       "early_assessments_passed", "first_half_vle_clicks"]
OULAD_CATS = {"gender", "region", "highest_education", "imd_band", "age_band", "disability",
              "code_module", "code_presentation"}
SHARED = ["gender_harmonized", "age_group_harmonized"]
ENSEMBLE = "Soft vote (LR + XGBoost)"
MODELS = ["Logistic regression", "XGBoost", "Random forest", "Linear SVM", "MLP", ENSEMBLE]


def make_data(original_path, oulad_path):
    orig = pd.read_csv(original_path)
    orig["target"] = (orig["Target"] == "Dropout").astype(int)
    orig["group"] = np.arange(len(orig))  # original file has one row per student
    orig["gender_harmonized"] = orig.Gender.map({0: "Female", 1: "Male", "0": "Female", "1": "Male"})
    orig["age_group_harmonized"] = pd.cut(
        pd.to_numeric(orig["Age at enrollment"], errors="coerce"),
        bins=[-np.inf, 34, 54, np.inf], labels=["under_35", "35_to_54", "55_plus"]
    ).astype("object")

    oulad = pd.read_csv(oulad_path)
    oulad["target"] = (oulad["final_result"] == "Withdrawn").astype(int)
    oulad["group"] = oulad["id_student"]
    oulad["gender_harmonized"] = oulad.gender.map({"F": "Female", "M": "Male"})
    oulad["age_group_harmonized"] = oulad.age_band.map(
        {"0-35": "under_35", "35-55": "35_to_54", "55<=": "55_plus", "<35": "under_35", "35-55": "35_to_54"}
    )
    return {"Original": orig, "OULAD": oulad}


def make_preprocessor(columns, categorical):
    cat = [c for c in columns if c in categorical]
    num = [c for c in columns if c not in categorical]
    transformers = []
    if cat:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat))
    if num:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num))
    return ColumnTransformer(transformers)


def build_model(name, y_train, seed=42):
    positives = max(int(np.sum(y_train)), 1)
    negatives = max(len(y_train) - positives, 1)
    if name == "Logistic regression":
        return LogisticRegression(class_weight="balanced", max_iter=1500, solver="liblinear", random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=220, max_depth=4, learning_rate=0.05,
                             subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
                             reg_lambda=1.0, scale_pos_weight=negatives / positives,
                             tree_method="hist", eval_metric="logloss", n_jobs=-1,
                             random_state=seed, verbosity=0)
    if name == "Random forest":
        return RandomForestClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                             alpha=1e-3, batch_size=512, learning_rate_init=0.001,
                             max_iter=120, early_stopping=True, n_iter_no_change=12,
                             validation_fraction=0.1, random_state=seed)
    return LinearSVC(class_weight="balanced", C=1.0, dual="auto", max_iter=5000, random_state=seed)


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metric_row(dataset, stage, model_name, y, prediction, score, split_type):
    return {"dataset": dataset, "stage": stage, "model": model_name, "split": split_type,
            "accuracy": accuracy_score(y, prediction),
            "dropout_recall": recall_score(y, prediction, zero_division=0),
            "dropout_precision": precision_score(y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y, score)}


def run_cv(data, name, stage, columns, categorical, splits, output, models):
    X, y = data[columns], data.target.to_numpy()
    rows = []
    for model_name in models:
        oof_pred = np.zeros(len(y), dtype=int)
        oof_score = np.zeros(len(y), dtype=float)
        start = time.time()
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            prep = make_preprocessor(columns, categorical)
            if model_name == ENSEMBLE:
                pipes = [Pipeline([("preprocess", make_preprocessor(columns, categorical)),
                                   ("model", build_model(base, y[train_idx], seed=42 + fold))])
                         for base in ("Logistic regression", "XGBoost")]
                for pipe in pipes:
                    pipe.fit(X.iloc[train_idx], y[train_idx])
                raw = np.mean([pipe.predict_proba(X.iloc[test_idx])[:, 1] for pipe in pipes], axis=0)
            else:
                estimator = build_model(model_name, y[train_idx], seed=42 + fold)
                pipe = Pipeline([("preprocess", prep), ("model", estimator)])
                if model_name == "MLP":
                    weights = compute_sample_weight(class_weight="balanced", y=y[train_idx])
                    pipe.fit(X.iloc[train_idx], y[train_idx], model__sample_weight=weights)
                else:
                    pipe.fit(X.iloc[train_idx], y[train_idx])
                raw = scores(pipe, X.iloc[test_idx])
            oof_score[test_idx] = raw
            oof_pred[test_idx] = (raw >= (0.0 if model_name == "Linear SVM" else 0.5)).astype(int)
            print(f"{name} {stage} | {model_name} | fold {fold}/{len(splits)} complete", flush=True)
        rows.append(metric_row(name, stage, model_name, y, oof_pred, oof_score, "5-fold OOF"))
        print(f"Completed {name} {stage} {model_name} in {time.time()-start:.1f}s", flush=True)
    return rows


def cross_domain(datasets, models):
    rows = []
    for source, target in [("Original", "OULAD"), ("OULAD", "Original")]:
        train, test = datasets[source], datasets[target]
        for model_name in models:
            if model_name == ENSEMBLE:
                pipes = [Pipeline([("preprocess", make_preprocessor(SHARED, set(SHARED))),
                                   ("model", build_model(base, train.target.to_numpy()))])
                         for base in ("Logistic regression", "XGBoost")]
                for pipe in pipes:
                    pipe.fit(train[SHARED], train.target)
                raw = np.mean([pipe.predict_proba(test[SHARED])[:, 1] for pipe in pipes], axis=0)
            else:
                prep = make_preprocessor(SHARED, set(SHARED))
                pipe = Pipeline([("preprocess", prep), ("model", build_model(model_name, train.target.to_numpy()))])
                if model_name == "MLP":
                    weights = compute_sample_weight(class_weight="balanced", y=train.target.to_numpy())
                    pipe.fit(train[SHARED], train.target, model__sample_weight=weights)
                else:
                    pipe.fit(train[SHARED], train.target)
                raw = scores(pipe, test[SHARED])
            pred = (raw >= (0.0 if model_name == "Linear SVM" else 0.5)).astype(int)
            rows.append(metric_row(f"Train {source} -> test {target}", "Shared features", model_name,
                                   test.target, pred, raw, "external dataset"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, default=Path("cleaned_data.csv"))
    ap.add_argument("--oulad", type=Path, default=Path("oulad_student_level.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/dataset_comparison"))
    ap.add_argument("--cross-only", action="store_true", help="recompute cross-dataset metrics without rerunning the slower CV experiments")
    ap.add_argument("--mlp-only", action="store_true", help="run just MLP and preserve existing model results")
    ap.add_argument("--ensemble-only", action="store_true", help="run just the LR + XGBoost soft-voting ensemble")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = make_data(args.original, args.oulad)
    if not args.cross_only:
        rows = []
        selected_models = (["MLP"] if args.mlp_only else
                           [ENSEMBLE] if args.ensemble_only else MODELS)
        folds = {}
        for name, data in datasets.items():
            splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                        if name == "OULAD" else StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
            folds[name] = list(splitter.split(np.zeros(len(data)), data.target, groups=data.group if name == "OULAD" else None))
            stages = [("Stage 1", ORIG_S1 if name == "Original" else OULAD_S1),
                      ("Stage 2", ORIG_S2 if name == "Original" else OULAD_S2)]
            cats = ORIG_CATS if name == "Original" else OULAD_CATS
            for stage, cols in stages:
                rows.extend(run_cv(data, name, stage, cols, cats, folds[name], args.output_dir, selected_models))
        cv = pd.DataFrame(rows)
        cv_path = args.output_dir / "within_dataset_cv.csv"
        if (args.mlp_only or args.ensemble_only) and cv_path.exists():
            previous = pd.read_csv(cv_path)
            replaced = selected_models[0]
            cv = pd.concat([previous.loc[previous.model != replaced], cv], ignore_index=True)
        cv.to_csv(args.output_dir / "within_dataset_cv.csv", index=False)
    else:
        cv_path = args.output_dir / "within_dataset_cv.csv"
        cv = pd.read_csv(cv_path) if cv_path.exists() else pd.DataFrame()
    external_models = (["MLP"] if args.mlp_only else
                       [ENSEMBLE] if args.ensemble_only else MODELS)
    external = pd.DataFrame(cross_domain(datasets, external_models))
    external_path = args.output_dir / "cross_dataset_shared_features.csv"
    if (args.mlp_only or args.ensemble_only) and external_path.exists():
        previous = pd.read_csv(external_path)
        replaced = external_models[0]
        external = pd.concat([previous.loc[previous.model != replaced], external], ignore_index=True)
    external.to_csv(external_path, index=False)
    config = {
        "folds": 5, "seed": 42, "original_target": "Dropout=1; Enrolled and Graduate=0",
        "oulad_target": "Withdrawn=1; Pass/Fail/Distinction=0",
        "models": MODELS,
        "mlp": "32/16 hidden units; balanced sample weights computed on training data only; early stopping",
        "ensemble": "equal soft-vote average of logistic-regression and XGBoost probabilities",
        "oulad_grouping": "all presentations for a student stay in one fold",
        "cross_dataset_features": SHARED,
        "cross_dataset_warning": "OULAD Withdrawn means withdrawn from a module-presentation, not necessarily university dropout; transfer results are exploratory and not directly equivalent.",
    }
    (args.output_dir / "experiment_setup.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    md = ["# Model comparison", "", "Within-dataset results use pooled out-of-fold predictions from stratified 5-fold CV (grouped by student for OULAD). Models use class weighting; MLP uses balanced sample weights computed on training folds only. The default decision threshold is 0.50 (zero decision score for linear SVM).", "",
          "## Within-dataset CV", "", cv.to_markdown(index=False, floatfmt=".3f"), "",
          "## Cross-dataset transfer", "", "These tests use only harmonized gender and approximate age band. OULAD Withdrawn is withdrawal from a module-presentation, not necessarily university dropout. Its outcome differs from the original institution's Dropout vs Enrolled/Graduate grouping, so transfer metrics are exploratory and not a direct validation of university dropout prediction.", "", external.to_markdown(index=False, floatfmt=".3f"), "",
          "## Interpretation", "", "Accuracy is reported, but recall is the priority. Stage 2 generally improves within-dataset discrimination. The soft-voting ensemble does not consistently beat the strongest individual model, so it should only be retained if a chosen operational metric benefits. Cross-dataset AUC is near chance with the limited shared features; there is no evidence of reliable transfer between these sources.", ""]
    (args.output_dir / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("Cross-dataset results:\n", external.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
