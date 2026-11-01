"""Prepare missing local model assets, then launch the Streamlit frontend."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "dataset_comparison"


def run(script, *args):
    print(f"\nPreparing project data with {script} ...", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)


def main():
    if (not (ROOT / "cleaned_data.csv").exists()
            or not (ROOT / "data_quality.json").exists()):
        run("clean_data.py", "data.csv", "--output", "cleaned_data.csv", "--report", "data_quality.json")
    if (not (RESULTS / "within_dataset_cv.csv").exists()
            or not (RESULTS / "cross_dataset_shared_features.csv").exists()):
        run("compare_datasets.py")
    expected_models = [
        RESULTS / "models" / f"{dataset}_stage_{stage}_soft_vote.joblib"
        for dataset in ("original", "oulad")
        for stage in (1, 2)
    ]
    if (not all(path.exists() for path in expected_models)
            or not (RESULTS / "selected_ensemble_setup.json").exists()
            or not (RESULTS / "selected_ensemble_metrics.csv").exists()):
        run("build_ensembles.py")
    expected_shap = [RESULTS / "shap" / f"{dataset}_stage_{stage}_{kind}.png"
                     for dataset in ("original", "oulad")
                     for stage in ("1", "2")
                     for kind in ("global", "local")]
    if not all(path.exists() for path in expected_shap):
        run("make_comparison_shap.py")
    print("\nStarting the website. Student predictions and model results appear in your browser.", flush=True)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
