# Student Dropout Early-Warning ML

A reproducible two-stage binary classification pipeline for the supplied student dataset. It trains only from the supplied data; no synthetic predictions or examples are included.

## Run

Requires Python 3.10+.

```powershell
py -m pip install -r requirements.txt
py clean_data.py data.csv --output cleaned_data.csv --report data_quality.json
py train_evaluate.py cleaned_data.csv --output-dir results
```

The training command uses one stratified 80/20 split (random seed 42) shared by both stages. It saves four fitted pipelines under `results/models`, metrics CSV/Markdown, stage-change tables, a split record, and SHAP bar plots for the random forest models. Threshold is 0.50 by default; use validation data to select a different threshold before deployment.

## Feature policy

- Stage 1: admission grade, previous qualification grade, age, course, application mode, attendance mode, scholarship holder, debtor, gender, and education special needs.
- Stage 2: the same fields plus first-semester units approved and first-semester grade.
- No second-semester fields are passed to either model. They describe a later outcome period and create leakage.
- Target is 1 for `Dropout` and 0 for both `Enrolled` and `Graduate`.
- Integer codes for course and application mode are one-hot encoded, not treated as ordered values. For official code definitions, use the [UCI dataset page and variable table](https://archive.ics.uci.edu/dataset/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess).

## Outputs

See `analysis.md` for results and limitations, `data_quality.json` for checks, `results/metrics_comparison.md` for the comparison, and `results/plots/` for SHAP images. The trained `.joblib` files are serialized Python objects; load only artifacts you trust.

## Launch the website

The project includes a Streamlit dashboard that loads the saved Stage 1 and Stage 2 random-forest pipelines, including their fitted preprocessing. It does not generate placeholder predictions. Install the requirements once, then start it:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

On Windows, you can double-click `start_dashboard.bat` to install dependencies and open the app. In the dashboard, use the rectangular **Student assessment** and **Model results** tabs. The student page asks for an institutional reference and labeled enrollment fields; choose whether the assessment is at enrollment or after first semester, then run it. The other tab contains the performance table and SHAP plots. The **Night mode** switch changes the page colors. Student entries are not saved by this prototype. Keep internet access available for the first install.
