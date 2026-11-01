from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "dataset_comparison"
MODELS = RESULTS / "models"
STAGE_FEATURES = {
    1: ["Admission grade", "Previous qualification (grade)", "Age at enrollment", "Course",
        "Daytime/evening attendance", "Scholarship holder", "Debtor",
        "Gender", "Education Special Needs"],
    2: ["Admission grade", "Previous qualification (grade)", "Age at enrollment", "Course",
        "Daytime/evening attendance", "Scholarship holder", "Debtor",
        "Gender", "Education Special Needs", "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)"],
}
COURSES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service — evening", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management — evening",
}
DISPLAY_NAMES = {
    "Admission grade": "Admission grade",
    "Previous qualification (grade)": "Prior qualification grade",
    "Age at enrollment": "Age at enrollment",
    "Course": "Course",
    "Daytime/evening attendance": "Attendance schedule",
    "Scholarship holder": "Scholarship holder",
    "Debtor": "Debtor status",
    "Gender": "Gender",
    "Education Special Needs": "Education special needs",
    "Curricular units 1st sem (approved)": "First-semester units approved",
    "Curricular units 1st sem (grade)": "First-semester average grade",
}

st.set_page_config(page_title="Student Support | Early Warning", page_icon="S", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --page:#fff8f1; --surface:#fffdfb; --text:#5b473e; --muted:#806a60; --line:#f2cbbb; --coral:#ff947b; --coral-dark:#e8755d; --peach:#ffd4c0; --peach-light:#ffe4d5; --orange:#ffad7d; }
[data-testid="stAppViewContainer"], [data-testid="stHeader"], header[data-testid="stHeader"] { background:var(--page); color:var(--text); }
.block-container { padding-top:1rem; max-width:1350px; }
.topbar { background:var(--coral); color:#573f36; padding:14px 20px; margin-bottom:18px; display:flex; align-items:center; justify-content:space-between; border-radius:16px; }
.topbrand { font-size:18px; font-weight:750; letter-spacing:.2px; }
.topdesc { color:#684d42; font-size:13px; }
.page-title { font-size:29px; line-height:1.2; font-weight:700; color:var(--text); margin:0 0 4px; }
.page-subtitle { color:var(--muted); margin:0 0 18px; font-size:14px; }
.section-title { font-size:18px; font-weight:700; color:var(--text); margin:8px 0 3px; }
.section-help { color:var(--muted); font-size:13px; margin:0 0 12px; }
.notice { background:var(--peach); border-left:5px solid var(--coral-dark); color:#5b473e; padding:12px 15px; font-size:13px; margin:8px 0 16px; border-radius:8px; }
.result-card { background:var(--peach-light); border:1px solid var(--line); border-left:6px solid var(--coral-dark); padding:16px 18px; margin-top:16px; border-radius:10px; }
.result-card.clear { border-left-color:#788b63; }
.small-copy { font-size:12px; color:var(--muted); line-height:1.55; }
[data-baseweb="tab-list"] { gap:7px; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { background:var(--peach); border:1px solid var(--line); border-radius:9px 9px 0 0; color:var(--text); padding:10px 18px; height:auto; }
button[data-baseweb="tab"][aria-selected="true"] { background:var(--surface); color:#9f4e3a; border-bottom:4px solid var(--coral-dark); font-weight:750; }
label, p, h1, h2, h3, [data-testid="stCaptionContainer"] { color:var(--text); }
[data-testid="stWidgetLabel"] p { color:var(--text) !important; font-weight:600; }
input, textarea, [data-baseweb="select"] > div { background:var(--surface) !important; color:var(--text) !important; }
[data-testid="stMetric"] { background:var(--peach-light); border:1px solid var(--line); padding:12px 14px; border-radius:10px; }
[data-testid="stDataFrame"], [data-testid="stForm"] { background:var(--surface); border-color:var(--line); }
.stButton button[kind="primary"], button[kind="primary"] { background:var(--coral-dark); border:1px solid var(--coral-dark); color:white; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

night = st.toggle("Night mode", value=False, key="night_mode")
if night:
    st.markdown("""
    <style>
    :root { --page:#211b19; --surface:#302723; --text:#fff1e8; --muted:#e0c6b7; --line:#65483d; --coral:#a95846; --coral-dark:#df8068; --peach:#49352d; --peach-light:#392d28; --orange:#bd7958; }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], header[data-testid="stHeader"] { background:var(--page) !important; color:var(--text) !important; }
    [data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stForm"] { background:var(--surface) !important; color:var(--text) !important; }
    button[data-baseweb="tab"] { background:#49352d !important; color:#fff1e8 !important; border-color:#65483d !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background:#302723 !important; color:#ffd4c0 !important; }
    label, p, h1, h2, h3, span, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:var(--text) !important; }
    input, textarea, [data-baseweb="select"] > div { background:#302723 !important; color:#fff1e8 !important; }
    .notice { color:var(--text); }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_ensemble(stage):
    return joblib.load(MODELS / f"original_stage_{stage}_soft_vote.joblib")


@st.cache_data
def load_students():
    return pd.read_csv(ROOT / "cleaned_data.csv")


@st.cache_data
def load_page_data():
    metrics = pd.read_csv(RESULTS / "within_dataset_cv.csv")
    transfer = pd.read_csv(RESULTS / "cross_dataset_shared_features.csv")
    with (RESULTS / "selected_ensemble_setup.json").open(encoding="utf-8") as f:
        setup = json.load(f)
    return metrics, transfer, setup


@st.cache_resource
def load_local_explainer(stage):
    import shap

    columns = STAGE_FEATURES[stage]
    model = load_ensemble(stage)
    background = load_students()[columns].sample(n=min(20, len(load_students())), random_state=42)

    def score_rows(values):
        frame = values.copy() if isinstance(values, pd.DataFrame) else pd.DataFrame(values, columns=columns)
        return model.predict_proba(frame[columns])[:, 1]

    masker = shap.maskers.Independent(background, max_samples=20)
    return shap.Explainer(score_rows, masker, algorithm="permutation")


def course_label(code):
    code = int(code)
    return f"{COURSES.get(code, 'Course')} ({code})"


def display_value(column, value):
    if column == "Course":
        return course_label(value)
    if column == "Daytime/evening attendance":
        return "Daytime" if int(value) == 1 else "Evening"
    if column in {"Scholarship holder", "Debtor", "Education Special Needs"}:
        return "Yes" if int(value) == 1 else "No"
    if column == "Gender":
        return "Male (source code 1)" if int(value) == 1 else "Female (source code 0)"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:g}"
    return str(value)


def show_shap_reason(row, stage, prefix):
    columns = STAGE_FEATURES[stage]
    values = pd.DataFrame([{c: row[c] for c in columns}], columns=columns)
    try:
        explainer = load_local_explainer(stage)
        explanation = explainer(values, max_evals=(2 * len(columns) + 1) * 3)
        contributions = np.asarray(explanation.values).reshape(-1)
        entries = []
        for feature, value, contribution in zip(columns, values.iloc[0].tolist(), contributions):
            entries.append({
                "Input": DISPLAY_NAMES.get(feature, feature),
                "Entered value": display_value(feature, value),
                "Effect on model score": "Raised risk score" if contribution > 0 else "Lowered risk score",
                "SHAP contribution": float(contribution),
            })
        detail = pd.DataFrame(entries).sort_values("SHAP contribution", key=lambda s: s.abs(), ascending=False)
        st.markdown(f"<div class='section-title'>{prefix}: why the model gave this score</div>", unsafe_allow_html=True)
        st.caption("SHAP compares this record with reference records and attributes parts of the ensemble score to its inputs. Positive values raise the score; negative values lower it. These are learned associations, not causes.")
        st.dataframe(detail.style.format({"SHAP contribution": "{:+.1%}"}), hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f"The model score was calculated, but its SHAP explanation could not be generated: {exc}")


required_assets = [
    ROOT / "cleaned_data.csv",
    RESULTS / "within_dataset_cv.csv",
    RESULTS / "cross_dataset_shared_features.csv",
    RESULTS / "selected_ensemble_setup.json",
    MODELS / "original_stage_1_soft_vote.joblib",
    MODELS / "original_stage_2_soft_vote.joblib",
]
if not all(path.exists() for path in required_assets):
    st.error("Project models are not prepared yet. Close this page and run `py run_app.py` from the project folder.")
    st.stop()

metrics, transfer_metrics, ensemble_setup = load_page_data()
students = load_students()
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "risk_result" not in st.session_state:
    st.session_state.risk_result = None

st.markdown("<div class='topbar'><div class='topbrand'>STUDENT SUPPORT &nbsp; / &nbsp; EARLY WARNING</div><div class='topdesc'>Staff decision-support prototype</div></div>", unsafe_allow_html=True)
st.markdown("<h1 class='page-title'>Student support review</h1><p class='page-subtitle'>Use the information available at the selected point in the academic year to guide a supportive check-in.</p>", unsafe_allow_html=True)

risk_tab, prediction_tab, results_tab = st.tabs(["Students at risk", "Prediction", "Model results"])

with risk_tab:
    st.markdown("<div class='notice'><b>Purpose:</b> screen a selected course cohort and review the highest model scores. No student names or IDs are present in the supplied dataset.</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        available_courses = sorted(pd.to_numeric(students.Course, errors="coerce").dropna().astype(int).unique())
        risk_course = st.selectbox("Course", available_courses, format_func=course_label, key="risk_course")
    with right:
        risk_period = st.radio("Information available", ["At enrollment", "After first semester"], horizontal=True, key="risk_period")
    risk_stage = 1 if risk_period == "At enrollment" else 2
    st.caption("Flags records scoring at or above the current 50% review threshold. The displayed record reference is only its row number in this anonymized dataset; it is not a real student ID.")
    if st.button("Screen this course", type="primary", key="screen_course"):
        cohort = students.loc[pd.to_numeric(students.Course, errors="coerce") == int(risk_course)].copy()
        model = load_ensemble(risk_stage)
        cohort["Dataset record"] = [f"REC-{i + 1:05d}" for i in cohort.index]
        cohort["Model score"] = model.predict_proba(cohort[STAGE_FEATURES[risk_stage]])[:, 1]
        cohort = cohort.sort_values("Model score", ascending=False)
        st.session_state.risk_result = {
            "course": int(risk_course), "stage": risk_stage,
            "cohort": cohort,
        }

    risk_result = st.session_state.risk_result
    if risk_result and risk_result["course"] == int(risk_course) and risk_result["stage"] == risk_stage:
        cohort = risk_result["cohort"]
        flagged = cohort.loc[cohort["Model score"] >= 0.5].copy()
        a, b = st.columns(2)
        a.metric("Records in course", len(cohort))
        b.metric("Flagged for staff review", len(flagged))
        if flagged.empty:
            st.success("No records reached the current review threshold.")
        else:
            table = flagged[["Dataset record", "Model score"]].copy()
            table["Model score"] = table["Model score"].map(lambda x: f"{x:.1%}")
            st.dataframe(table, hide_index=True, use_container_width=True)
            chosen_ref = st.selectbox("Select a flagged record to inspect its reasoning", flagged["Dataset record"].tolist(), key="risk_reason_record")
            selected = flagged.loc[flagged["Dataset record"] == chosen_ref].iloc[0]
            st.markdown(f"<div class='result-card'><b>{chosen_ref}</b> · {course_label(risk_course)} · model score <b>{selected['Model score']:.1%}</b></div>", unsafe_allow_html=True)
            show_shap_reason(selected, risk_stage, chosen_ref)
    elif risk_result:
        st.info("Your course or assessment point changed. Click **Screen this course** to refresh the list.")

with prediction_tab:
    st.markdown("<div class='notice'><b>Individual assessment:</b> enter a student's record to receive an ensemble score and a SHAP explanation of which inputs raised or lowered it.</div>", unsafe_allow_html=True)
    period = st.radio("Assessment point", ["At enrollment", "After first semester"], horizontal=True, key="prediction_period")
    stage = 1 if period == "At enrollment" else 2
    with st.form(f"prediction_form_{stage}"):
        student_ref = st.text_input("Student reference (optional)", placeholder="Your institution's reference, if available")
        st.markdown("<div class='section-title'>Enrollment and admission information</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            admission = st.number_input("Admission grade (0–200)", 0.0, 200.0, 126.1, .1)
            previous_grade = st.number_input("Previous qualification grade (0–200)", 0.0, 200.0, 133.1, .1)
            age = st.number_input("Age at enrollment", 16, 100, 20, 1)
            course_codes = sorted(pd.to_numeric(students.Course, errors="coerce").dropna().astype(int).unique())
            course = st.selectbox("Course", course_codes, format_func=course_label, key=f"prediction_course_{stage}")
        with c2:
            attendance = st.selectbox("Attendance schedule", [1, 0], format_func=lambda x: "Daytime" if x == 1 else "Evening")
            gender = st.selectbox("Gender as coded in source data", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            scholarship = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            debtor = st.selectbox("Debtor status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            special = st.selectbox("Education special needs recorded", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        if stage == 2:
            st.markdown("<div class='section-title'>First-semester academic information</div><div class='section-help'>Second-semester information is not used.</div>", unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                approved = st.number_input("First-semester curricular units approved", 0, 26, 5, 1)
            with g2:
                semester_grade = st.number_input("First-semester average grade (0–20)", 0.0, 20.0, 12.3, .1)
        submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

    if submitted:
        row = {
            "Admission grade": admission,
            "Previous qualification (grade)": previous_grade,
            "Age at enrollment": age,
            "Course": int(course),
            "Daytime/evening attendance": int(attendance),
            "Scholarship holder": int(scholarship),
            "Debtor": int(debtor),
            "Gender": int(gender),
            "Education Special Needs": int(special),
        }
        if stage == 2:
            row["Curricular units 1st sem (approved)"] = int(approved)
            row["Curricular units 1st sem (grade)"] = float(semester_grade)
        score = float(load_ensemble(stage).predict_proba(pd.DataFrame([row]))[0, 1])
        st.session_state.prediction_result = {"stage": stage, "row": row, "score": score, "reference": student_ref.strip()}

    result = st.session_state.prediction_result
    if result and result["stage"] == stage:
        score = result["score"]
        flag = score >= 0.5
        label = "Review for supportive follow-up" if flag else "Below the current review threshold"
        card = "result-card" if flag else "result-card clear"
        st.markdown(f"<div class='{card}'><div class='section-title'>{label}</div><p>Student reference: <b>{result['reference'] or 'Not supplied'}</b></p><p>Ensemble model score: <b>{score:.1%}</b> &nbsp; | &nbsp; Review threshold: <b>50%</b></p></div>", unsafe_allow_html=True)
        st.progress(score)
        show_shap_reason(result["row"], stage, "Individual assessment")
        st.caption("This score is not calibrated and is not a confirmed probability. SHAP explains how the model's inputs contributed to its score; it does not prove why a student may leave.")

with results_tab:
    st.markdown("<div class='section-title'>Model evaluation</div><div class='section-help'>Cross-validation results are separate from individual student assessments.</div>", unsafe_allow_html=True)
    dataset_label = st.selectbox("Dataset", ["Original university dataset", "OULAD course dataset"], key="evaluation_dataset")
    dataset_key = "Original" if dataset_label.startswith("Original") else "OULAD"
    point = st.radio("Prediction point", ["At enrollment", "After first-semester information"], horizontal=True, key="evaluation_stage")
    stage_key = "Stage 1" if point == "At enrollment" else "Stage 2"
    rows = metrics.loc[(metrics.dataset == dataset_key) & (metrics.stage == stage_key)].copy()
    if dataset_key == "OULAD":
        st.caption("OULAD's positive outcome is withdrawal from a module presentation, not necessarily university dropout.")
        positive_name = "Module withdrawal"
    else:
        st.caption("The original dataset groups Enrolled and Graduate as non-dropout. All metrics are five-fold out-of-fold estimates.")
        positive_name = "Dropout"
    members = ensemble_setup["members_by_dataset_stage"][f"{dataset_key} {stage_key}"]["members"]
    st.info(f"Selected ensemble: {' + '.join(members)}")
    if not rows.empty:
        renamed = rows.rename(columns={"model": "Model", "accuracy": "Accuracy", "dropout_recall": f"{positive_name} recall", "dropout_precision": f"{positive_name} precision", "roc_auc": "ROC-AUC"})
        cols = ["Model", "Accuracy", f"{positive_name} recall", f"{positive_name} precision", "ROC-AUC"]
        st.dataframe(renamed[cols].sort_values(f"{positive_name} recall", ascending=False).style.format({c: "{:.3f}" for c in cols[1:]}), hide_index=True, use_container_width=True)
        st.bar_chart(renamed.set_index("Model")[[f"{positive_name} recall", "ROC-AUC"]], use_container_width=True)
    else:
        st.error("No model results were found. Close the app and run it again with run_app.py.")

    with st.expander("Cross-dataset comparison"):
        st.caption("Only gender and approximate age band were aligned. Results are exploratory; the two datasets have different outcome definitions.")
        transfer = transfer_metrics.rename(columns={"dataset": "Train → test", "model": "Model", "accuracy": "Accuracy", "dropout_recall": "Positive-class recall", "dropout_precision": "Positive-class precision", "roc_auc": "ROC-AUC"})
        st.dataframe(transfer[["Train → test", "Model", "Accuracy", "Positive-class recall", "Positive-class precision", "ROC-AUC"]].style.format({c: "{:.3f}" for c in ["Accuracy", "Positive-class recall", "Positive-class precision", "ROC-AUC"]}), hide_index=True, use_container_width=True)

    st.markdown("<div class='section-title'>SHAP model explanations</div><div class='section-help'>The plots summarize the selected individual model. Per-record explanations above use SHAP on the actual ensemble prediction.</div>", unsafe_allow_html=True)
    selected_models = {("Original", "Stage 1"): "Linear SVM", ("Original", "Stage 2"): "Logistic regression", ("OULAD", "Stage 1"): "Logistic regression", ("OULAD", "Stage 2"): "XGBoost"}
    slug = f"{dataset_key.lower()}_{stage_key.lower().replace(' ', '_')}"
    shap_dir = RESULTS / "shap"
    x1, x2 = st.columns(2)
    with x1:
        st.image(shap_dir / f"{slug}_global.png", caption=f"Global contributions ({selected_models[(dataset_key, stage_key)]})")
    with x2:
        st.image(shap_dir / f"{slug}_local.png", caption=f"Held-out example ({selected_models[(dataset_key, stage_key)]})")
    st.caption("SHAP shows learned associations, not causes. Model scores are not calibrated probabilities.")

st.markdown("<hr><p class='small-copy'>Research prototype · Student details entered in the Prediction tab are not saved. Dataset record numbers are anonymous row references, not real student identifiers.</p>", unsafe_allow_html=True)
