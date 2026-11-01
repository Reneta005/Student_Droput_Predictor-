"""Create a student-level, early-warning table from the linked OULAD CSV tables.

Usage: py consolidate_oulad.py <folder-containing-oulad-csv-files> [output.csv]

The output is OULAD-only. It deliberately preserves OULAD's original outcome
labels and does not claim that they are identical to the other university's
Dropout/Enrolled/Graduate labels.
"""
from pathlib import Path
import argparse
import pandas as pd


def consolidate(folder: Path) -> pd.DataFrame:
    info = pd.read_csv(folder / "studentInfo.csv", na_values="?")
    registrations = pd.read_csv(folder / "studentRegistration.csv", na_values="?")
    courses = pd.read_csv(folder / "courses.csv", na_values="?")
    assessments = pd.read_csv(folder / "assessments.csv", na_values="?")
    submissions = pd.read_csv(folder / "studentAssessment.csv", na_values="?")

    keys = ["code_module", "code_presentation"]
    table = info.merge(courses, on=keys, how="left", validate="many_to_one")
    table = table.merge(registrations, on=keys + ["id_student"], how="left", validate="one_to_one")

    # Treat the first half of each presentation as the early-warning window.
    assessments = assessments.merge(courses, on=keys, how="left", validate="many_to_one")
    assessments["early_window"] = assessments["date"] <= assessments["module_presentation_length"] / 2
    early_assessments = assessments.loc[assessments.early_window].copy()
    early_assessments = early_assessments.merge(submissions, on="id_assessment", how="left", validate="one_to_many")
    early_assessments = early_assessments.loc[early_assessments.id_student.notna()]
    early_assessments["score"] = pd.to_numeric(early_assessments["score"], errors="coerce")
    early_assessments["passed_assessment"] = (early_assessments.score >= 40).astype("int8")
    a = early_assessments.groupby(keys + ["id_student"], as_index=False).agg(
        early_assessments_submitted=("score", "count"),
        early_mean_assessment_score=("score", "mean"),
        early_assessments_passed=("passed_assessment", "sum"),
    )
    table = table.merge(a, on=keys + ["id_student"], how="left", validate="one_to_one")

    # The VLE file is large; stream it and retain only first-half activity.
    click_totals = {}
    lengths = courses.set_index(keys)["module_presentation_length"].to_dict()
    for chunk in pd.read_csv(folder / "studentVle.csv", na_values="?", chunksize=500_000):
        chunk["module_length"] = [lengths.get((m, p)) for m, p in zip(chunk.code_module, chunk.code_presentation)]
        chunk = chunk.loc[chunk.date <= chunk.module_length / 2]
        if chunk.empty:
            continue
        grouped = chunk.groupby(keys + ["id_student"], as_index=False)["sum_click"].sum()
        for row in grouped.itertuples(index=False, name=None):
            k = tuple(row[:3])
            click_totals[k] = click_totals.get(k, 0) + int(row[3])
    table["first_half_vle_clicks"] = [
        click_totals.get((m, p, sid), 0)
        for m, p, sid in zip(table.code_module, table.code_presentation, table.id_student)
    ]

    table["early_assessments_submitted"] = table.early_assessments_submitted.fillna(0).astype(int)
    table["early_assessments_passed"] = table.early_assessments_passed.fillna(0).astype(int)
    table["early_mean_assessment_score"] = table.early_mean_assessment_score.fillna(0)
    table["withdrawn_target"] = (table.final_result == "Withdrawn").astype(int)
    # Remove the non-model target text? Keep it for auditability and interpretation.
    # Exclude any later-performance measures: only first-half aggregates are included.
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=Path("oulad_student_level.csv"))
    args = parser.parse_args()
    result = consolidate(args.data_folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} student-presentation rows x {len(result.columns)} columns to {args.output}")
    print("Outcome counts:")
    print(result.final_result.value_counts(dropna=False).to_string())
    print(f"Duplicate student-presentation keys: {result.duplicated(['code_module','code_presentation','id_student']).sum()}")


if __name__ == "__main__":
    main()
