# Results and interpretation

## Framing

This is a two-stage early-warning experiment on the supplied higher-education dataset. Stage 1 estimates risk using enrollment-time fields. Stage 2 re-estimates risk after first-semester results are known. These are risk flags for prioritizing human support, not determinations about an individual student.

## Data checks

The supplied CSV contains 4,424 students, 36 inputs, and one target. Its whole header row was quoted as a single field, so the cleaning script parses and trims that line explicitly, removes the BOM, normalizes `Nacionality` and `Educational special needs`, and verifies the expected 37-column schema. It found 0 duplicate rows and 0 missing cells. Labels: 1,421 Dropout, 794 Enrolled, 2,209 Graduate. The specified binary flags contain only 0/1. Category IDs are treated as categories by one-hot encoding; consult the UCI variable table for their official meanings.

## Evaluation

One stratified 80/20 train/test split with random seed 42 was reused for all four models. Results below use probability threshold 0.50 and class weighting. Dropout is the positive class.

| Stage | Model | Accuracy | Dropout recall | Dropout precision | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Stage 1 | Logistic regression | 0.708 | 0.725 | 0.534 | 0.786 |
| Stage 1 | Random forest | 0.722 | 0.609 | 0.562 | 0.781 |
| Stage 2 | Logistic regression | 0.805 | 0.743 | 0.678 | 0.864 |
| Stage 2 | Random forest | 0.837 | 0.708 | 0.767 | 0.883 |

Adding first-semester approved units and grade increased logistic regression accuracy by 0.096, dropout recall by 0.018, precision by 0.145, and ROC-AUC by 0.077. For random forest, the corresponding gains were 0.115, 0.099, 0.205, and 0.102. At this fixed threshold, Stage 2 random forest has the highest accuracy, precision, and ROC-AUC, while Stage 2 logistic regression has higher dropout recall (0.743 vs 0.708). Because recall is the priority, threshold selection should be tuned on validation data to the institution's acceptable follow-up capacity; this run does not claim the default 0.50 threshold is optimal.

## Explanations

The SHAP bar plots in `results/plots/` show mean absolute SHAP values for Stage 1 and Stage 2 random forests on up to 120 held-out test rows. They describe model associations and do not establish causes of dropout. The feature set includes debtor status, which can change during a semester and may encode conditions that institutions can influence.

## Leakage and limitations

Second-semester fields are excluded from both stages because they occur after the intended early-warning decision point and may reveal the eventual outcome. `Enrolled` and `Graduate` are grouped as non-dropout according to the requested definition; this is a modeling choice that combines students with different eventual statuses. Results come from a single random split of one institution's dataset, without external or later-cohort validation, calibration analysis, uncertainty intervals, or threshold tuning. They should not be interpreted as deployment-ready performance. Historical patterns and input quality may not generalize to another institution or cohort.
