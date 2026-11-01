# Student Dropout Early-Warning Project

This project compares models that estimate dropout risk at enrollment and after first-semester information is available. It includes a Streamlit website for staff-style assessment and model comparison.

## Run it

On Windows, double-click `start_dashboard.bat`.

Or open PowerShell in this folder and run:

```powershell
py -m pip install -r requirements.txt
py run_app.py
```

The browser opens the website with three tabs:

- **Students at risk:** choose a course and enrollment or first-semester information to list records above the review threshold. Select a record to see its SHAP explanation.
- **Prediction:** enter one student's information to get an ensemble score and a SHAP explanation of which inputs raised or lowered it.
- **Model results:** see cross-validation comparisons, ensemble members, the dataset comparison, and global SHAP plots.

The first run can take several minutes while models and explanations are prepared. Later runs start faster.

## What to upload to GitHub

Upload this project folder. Keep these files in the repository:

- Python scripts (`.py`), `README.md`, `requirements.txt`, and `start_dashboard.bat`
- `data.csv` and `oulad_student_level.csv` (the two input datasets needed to run the project)

The `.gitignore` file excludes generated models, reports, caches, and temporary files. The app recreates those locally when someone runs it, so you do not need to upload large model files.

## Push changes later

Using GitHub Desktop: add this local folder as a repository, publish it to your GitHub account, then whenever you change code, write a short commit message and click **Commit to main** followed by **Push origin**. The `.gitignore` keeps generated outputs out of the upload.

## Project notes

- Stage 1 uses enrollment-time information. Stage 2 adds first-semester academic information. No second-semester fields are used.
- The original dataset target is Dropout versus Enrolled/Graduate. OULAD's `Withdrawn` outcome means withdrawal from a module presentation, not necessarily leaving university. The datasets are analyzed separately; cross-dataset transfer is exploratory.
- Application mode was removed from the model and student form because its values describe Portuguese admissions routes and do not map meaningfully to Indian admissions. The original source CSV is kept intact, but this field is not used by either stage.
- The original dataset has no student names or IDs. At-risk list references are anonymous dataset row numbers, not real identities; use the view to demonstrate screening, not to contact actual students.
- Model scores are screening aids, not diagnoses or calibrated probabilities. SHAP describes model associations, not causes.
- `consolidate_oulad.py` documents how the linked OULAD tables were combined. The included `oulad_student_level.csv` is sufficient to run the app.
