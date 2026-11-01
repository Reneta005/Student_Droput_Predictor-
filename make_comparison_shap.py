"""Generate SHAP global and one-record explanations for recall-focused models.

Uses the first stratified fold from the model comparison experiment, so the
explained records are held out from model fitting. Outputs are model-score
contributions, not causal claims or calibrated individual probabilities.
"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from compare_datasets import (
    ORIG_S1, ORIG_S2, ORIG_CATS, OULAD_S1, OULAD_S2, OULAD_CATS,
    make_data, make_preprocessor, build_model,
)


BASE = Path(__file__).parent
OUT = BASE / "results" / "dataset_comparison" / "shap"
SELECTION = {
    ("Original", "Stage 1"): "Linear SVM",
    ("Original", "Stage 2"): "Logistic regression",
    ("OULAD", "Stage 1"): "Logistic regression",
    ("OULAD", "Stage 2"): "XGBoost",
}
DISPLAY = {
    "Admission grade": "Admission grade",
    "Previous qualification (grade)": "Prior qualification grade",
    "Age at enrollment": "Age at entry",
    "Daytime/evening attendance": "Attendance schedule",
    "Scholarship holder": "Scholarship holder",
    "Debtor": "Debtor status",
    "Gender": "Gender",
    "Education Special Needs": "Special needs recorded",
    "Curricular units 1st sem (approved)": "First-semester units approved",
    "Curricular units 1st sem (grade)": "First-semester average grade",
    "gender": "Gender",
    "region": "Region",
    "highest_education": "Previous education",
    "imd_band": "Area deprivation band",
    "age_band": "Age band",
    "num_of_prev_attempts": "Previous attempts",
    "studied_credits": "Credits registered",
    "disability": "Disability recorded",
    "code_module": "Module code",
    "code_presentation": "Presentation code",
    "early_assessments_submitted": "Early assessments submitted",
    "early_mean_assessment_score": "Mean early assessment score",
    "early_assessments_passed": "Early assessments passed",
    "first_half_vle_clicks": "First-half online activity clicks",
}


def pretty_feature_names(names, columns, categorical):
    result = []
    ordered_cats = sorted(categorical.intersection(columns), key=len, reverse=True)
    for raw in names:
        name = raw.removeprefix("numeric__").removeprefix("categorical__")
        onehot = False
        for col in ordered_cats:
            prefix = f"{col}_"
            if name.startswith(prefix):
                name = f"{DISPLAY.get(col, col)} = {name[len(prefix):]}"
                onehot = True
                break
        if not onehot:
            name = DISPLAY.get(name, name.replace("_", " "))
        result.append(name)
    return result


def binary_shap_values(explainer, matrix, model_name):
    if model_name == "XGBoost":
        values = explainer.shap_values(matrix)
        base = explainer.expected_value
    else:
        explanation = explainer(matrix)
        values = explanation.values
        base = explanation.base_values
    if isinstance(values, list):
        values = values[-1]
    elif np.ndim(values) == 3:
        values = values[:, :, -1]
    values = np.asarray(values)
    if np.ndim(base) > 0:
        base = np.asarray(base).reshape(-1)[-1]
    return values, float(base)


def group_shap_values(values, transformed_names, columns, categorical):
    """Add one-hot contributions back together into their original input fields."""
    grouped = []
    for column in columns:
        if column in categorical:
            prefix = f"categorical__{column}_"
            positions = [i for i, name in enumerate(transformed_names) if name.startswith(prefix)]
        else:
            target = f"numeric__{column}"
            positions = [i for i, name in enumerate(transformed_names) if name == target]
        grouped.append(values[:, positions].sum(axis=1) if positions else np.zeros(values.shape[0]))
    return np.column_stack(grouped)


def one_explanation(dataset, stage, data, columns, categorical):
    y = data.target.to_numpy()
    splitter = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                if dataset == "OULAD" else StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    train_idx, held_idx = next(splitter.split(
        np.zeros(len(data)), y, groups=data.group.to_numpy() if dataset == "OULAD" else None
    ))
    model_name = SELECTION[(dataset, stage)]
    pre = make_preprocessor(columns, categorical)
    x_train = pre.fit_transform(data.iloc[train_idx][columns])
    x_held = pre.transform(data.iloc[held_idx][columns])
    x_train = x_train.toarray() if hasattr(x_train, "toarray") else np.asarray(x_train)
    x_held = x_held.toarray() if hasattr(x_held, "toarray") else np.asarray(x_held)
    estimator = build_model(model_name, y[train_idx], seed=42)
    estimator.fit(x_train, y[train_idx])
    transformed_names = pre.get_feature_names_out()
    names = [DISPLAY.get(col, col.replace("_", " ")) for col in columns]

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(x_train), size=min(100, len(x_train)), replace=False)
    n_explain = min(250, len(x_held))
    ex_idx = rng.choice(len(x_held), size=n_explain, replace=False)
    background, sample = x_train[bg_idx], x_held[ex_idx]
    explainer = (shap.TreeExplainer(estimator) if model_name == "XGBoost"
                 else shap.LinearExplainer(estimator, background))
    values, base = binary_shap_values(explainer, sample, model_name)
    grouped_values = group_shap_values(values, transformed_names, columns, categorical)

    slug = f"{dataset.lower()}_{stage.lower().replace(' ', '_')}"
    OUT.mkdir(parents=True, exist_ok=True)
    importance = np.mean(np.abs(grouped_values), axis=0)
    order = np.argsort(importance)[-12:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.asarray(names)[order], importance[order], color="#d96c56")
    ax.set_xlabel("Mean absolute SHAP value (model-score impact)")
    ax.set_title(f"{dataset} {stage}: {model_name} global SHAP")
    fig.tight_layout()
    fig.savefig(OUT / f"{slug}_global.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Show one high-score held-out record as an example, not as a cause.
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x_held)[:, 1]
    else:
        scores = estimator.decision_function(x_held)
    chosen = int(np.argmax(scores))
    chosen_position = np.where(ex_idx == chosen)[0]
    if len(chosen_position) == 0:
        ex_idx = np.append(ex_idx, chosen)
        sample = x_held[ex_idx]
        values, base = binary_shap_values(explainer, sample, model_name)
        grouped_values = group_shap_values(values, transformed_names, columns, categorical)
        chosen_position = np.array([len(ex_idx) - 1])
    pos = int(chosen_position[0])
    explanation = shap.Explanation(
        values=grouped_values[pos], base_values=base,
        data=data.iloc[held_idx[ex_idx[pos]]][columns].to_numpy(), feature_names=names,
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    plt.title(f"{dataset} {stage}: held-out high-score example ({model_name})")
    plt.tight_layout()
    fig.savefig(OUT / f"{slug}_local.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(OUT / f"{slug}_importance.csv", index=False)
    print(f"{dataset} {stage}: {model_name}; held-out score={scores[chosen]:.4f}; "
          f"base={base:.4f}; saved global and local plots", flush=True)


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
    datasets = make_data(BASE / "cleaned_data.csv", BASE / "oulad_student_level.csv")
    for dataset, data in datasets.items():
        stages = ([("Stage 1", ORIG_S1, ORIG_CATS), ("Stage 2", ORIG_S2, ORIG_CATS)]
                  if dataset == "Original" else
                  [("Stage 1", OULAD_S1, OULAD_CATS), ("Stage 2", OULAD_S2, OULAD_CATS)])
        for stage, columns, categorical in stages:
            one_explanation(dataset, stage, data, columns, categorical)
    print(f"Saved SHAP outputs to {OUT}")


if __name__ == "__main__":
    main()
