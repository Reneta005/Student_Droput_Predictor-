from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
COURSES = {33:"Biofuel Production Technologies",171:"Animation and Multimedia Design",8014:"Social Service — evening",9003:"Agronomy",9070:"Communication Design",9085:"Veterinary Nursing",9119:"Informatics Engineering",9130:"Equinculture",9147:"Management",9238:"Social Service",9254:"Tourism",9500:"Nursing",9556:"Oral Hygiene",9670:"Advertising and Marketing Management",9773:"Journalism and Communication",9853:"Basic Education",9991:"Management — evening"}
APPLICATIONS = {1:"1st phase — general contingent",2:"Ordinance No. 612/93",5:"1st phase — Azores special contingent",7:"Holder of another higher-education course",10:"Ordinance No. 854-B/99",15:"International student (bachelor's)",16:"1st phase — Madeira special contingent",17:"2nd phase — general contingent",18:"3rd phase — general contingent",26:"Ordinance No. 533-A/99 (different plan)",27:"Ordinance No. 533-A/99 (other institution)",39:"Over 23 years old",42:"Transfer",43:"Change of course",44:"Technological specialization diploma holder",51:"Change of institution/course",53:"Short-cycle diploma holder",57:"International change of institution/course"}

st.set_page_config(page_title="Student Support | Early Warning", page_icon="S", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --page:#f3f5f7; --surface:#ffffff; --text:#202b38; --muted:#566575; --line:#d5dce4; --navy:#193f70; --blue:#2368a8; --soft:#e8eef5; --input:#ffffff; }
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:var(--page); color:var(--text); }
[data-testid="stSidebar"] { background:var(--surface); }
.block-container { padding-top:1.1rem; max-width:1320px; }
header[data-testid="stHeader"] { background:var(--page); }
.topbar { background:var(--navy); color:#fff; padding:12px 18px; margin-bottom:20px; display:flex; align-items:center; justify-content:space-between; border-radius:3px; }
.topbrand { font-size:18px; font-weight:700; letter-spacing:.2px; }
.topdesc { color:#e9f0f8; font-size:12px; }
.page-title { font-size:27px; line-height:1.2; font-weight:650; color:var(--text); margin:0 0 4px; }
.page-subtitle { color:var(--muted); margin:0 0 18px; font-size:14px; }
.section-title { font-size:17px; font-weight:650; color:var(--text); margin:8px 0 2px; }
.section-help { color:var(--muted); font-size:12px; margin:0 0 12px; }
.notice { background:var(--soft); border-left:4px solid var(--blue); color:var(--text); padding:11px 14px; font-size:13px; margin:8px 0 16px; }
.result-card { background:var(--surface); border:1px solid var(--line); border-left:5px solid #b7791f; padding:16px 18px; margin-top:18px; }
.result-card.clear { border-left-color:#25805a; }
.small-copy { font-size:12px; color:var(--muted); line-height:1.5; }
[data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { background:var(--soft); border:1px solid var(--line); border-radius:4px 4px 0 0; color:var(--text); padding:9px 18px; height:auto; }
button[data-baseweb="tab"][aria-selected="true"] { background:var(--surface); color:var(--navy); border-bottom:3px solid var(--blue); font-weight:700; }
label, p, h1, h2, h3, [data-testid="stCaptionContainer"] { color:var(--text); }
[data-testid="stWidgetLabel"] p { color:var(--text) !important; font-weight:550; }
input, textarea, [data-baseweb="select"] > div { background:var(--input) !important; color:var(--text) !important; }
[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); padding:12px 14px; }
[data-testid="stDataFrame"] { background:var(--surface); }
.stButton button[kind="primary"], button[kind="primary"] { background:var(--navy); border:1px solid var(--navy); border-radius:4px; }
[data-testid="stForm"] { background:var(--surface); border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

night = st.toggle("Night mode", value=False, key="night_mode")
if night:
    st.markdown("""
    <style>
    :root { --page:#151a20; --surface:#202832; --text:#eef2f6; --muted:#c0cad5; --line:#46515e; --navy:#102e50; --blue:#76b9f2; --soft:#2c3743; --input:#28333e; }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], header[data-testid="stHeader"] { background:var(--page) !important; color:var(--text) !important; }
    [data-testid="stSidebar"], [data-testid="stForm"], [data-testid="stMetric"], [data-testid="stDataFrame"] { background:var(--surface) !important; color:var(--text) !important; }
    .topbar { background:#102e50; }
    button[data-baseweb="tab"] { background:#2c3743 !important; color:#eef2f6 !important; border-color:#46515e !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background:#202832 !important; color:#a9d6ff !important; }
    label, p, h1, h2, h3, span, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:var(--text) !important; }
    input, textarea, [data-baseweb="select"] > div { background:#28333e !important; color:#eef2f6 !important; }
    [data-testid="stMarkdownContainer"] a { color:#9bcfff !important; }
    .notice { background:#2c3743; color:#eef2f6; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_forest(stage):
    return joblib.load(MODELS / f"stage_{stage}_random_forest.joblib")

@st.cache_data
def load_reference_files():
    with (ROOT / "data_quality.json").open(encoding="utf-8") as f:
        quality = json.load(f)
    metrics = pd.read_csv(ROOT / "results" / "metrics_comparison.csv")
    return quality, metrics

quality, metrics = load_reference_files()
def get_metric(stage, model, name):
    found = metrics[(metrics.stage == stage) & (metrics.model == model)]
    return float(found.iloc[0][name]) if not found.empty else 0.0

st.markdown("<div class='topbar'><div class='topbrand'>STUDENT SUPPORT &nbsp; / &nbsp; EARLY WARNING</div><div class='topdesc'>Staff decision-support prototype</div></div>", unsafe_allow_html=True)
st.markdown("<h1 class='page-title'>Student support review</h1><p class='page-subtitle'>Use the information available at the selected point in the academic year to guide a supportive check-in.</p>", unsafe_allow_html=True)

assessment_tab, results_tab = st.tabs(["Student assessment", "Model results"])

with assessment_tab:
    st.markdown("<div class='notice'><b>Purpose:</b> this tool helps staff identify students who may benefit from a conversation or additional support. It does not make decisions about students.</div>", unsafe_allow_html=True)
    stage_label = st.selectbox("When is this assessment being made?", ["At enrollment", "After first semester"], key="prediction_stage")
    stage = 1 if stage_label == "At enrollment" else 2
    with st.form(f"assessment_form_{stage}"):
        st.markdown("<div class='section-title'>Student reference</div><div class='section-help'>Use your institution's reference number to identify the record during this session. This prototype does not save student entries.</div>", unsafe_allow_html=True)
        student_ref = st.text_input("Student ID or reference number", placeholder="Enter the reference used by your institution")
        st.markdown("<div class='section-title'>Enrollment and admission</div><div class='section-help'>Enter values from the student's enrollment or admissions record.</div>", unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            admission=st.number_input("Admission grade (0–200)",0.0,200.0,126.1,.1,help="The student's admission grade. The source cohort ranged from 95 to 190.")
            previous_grade=st.number_input("Previous qualification grade (0–200)",0.0,200.0,133.1,.1,help="Grade for the qualification completed before enrollment.")
            age=st.number_input("Age when enrolled",16,100,20,1)
            course_codes=sorted(quality["unique_values"]["Course"])
            course=st.selectbox("Degree course",course_codes,format_func=lambda code:f"{COURSES.get(int(code), 'Other course')} ({code})")
            app_codes=sorted(quality["unique_values"]["Application mode"])
            application=st.selectbox("Admission route",app_codes,format_func=lambda code:f"{APPLICATIONS.get(int(code), 'Other application route')} ({code})")
        with c2:
            attendance=st.selectbox("Attendance schedule",[1,0],format_func=lambda code:"Daytime" if code==1 else "Evening")
            gender=st.selectbox("Gender in source record",[0,1],format_func=lambda code:"Female" if code==0 else "Male")
            scholarship=st.selectbox("Scholarship holder",[0,1],format_func=lambda code:"No" if code==0 else "Yes")
            debtor=st.selectbox("Debtor status",[0,1],format_func=lambda code:"No outstanding debt (0)" if code==0 else "Outstanding debt (1)")
            special=st.selectbox("Education special needs recorded",[0,1],format_func=lambda code:"No" if code==0 else "Yes")
        if stage == 2:
            st.markdown("<div class='section-title'>First-semester academic results</div><div class='section-help'>Use the completed first-semester record. No second-semester information is used.</div>", unsafe_allow_html=True)
            c3,c4=st.columns(2)
            with c3: approved=st.number_input("First-semester curricular units approved",0,26,5,1)
            with c4: semester_grade=st.number_input("First-semester average grade (0–20)",0.0,20.0,12.3,.1)
        submitted=st.form_submit_button("Run assessment",type="primary",use_container_width=True)

    if submitted:
        if not student_ref.strip():
            st.error("Enter a student ID or reference number so the result can be matched to the correct record.")
        else:
            row={"Admission grade":admission,"Previous qualification (grade)":previous_grade,"Age at enrollment":age,"Course":course,"Application mode":application,"Daytime/evening attendance":attendance,"Scholarship holder":scholarship,"Debtor":debtor,"Gender":gender,"Education Special Needs":special}
            if stage==2: row.update({"Curricular units 1st sem (approved)":approved,"Curricular units 1st sem (grade)":semester_grade})
            try:
                score=float(load_forest(stage).predict_proba(pd.DataFrame([row]))[0,1])
                review=score>=.5
                label="Review for supportive follow-up" if review else "Below the current review threshold"
                box="result-card" if review else "result-card clear"
                st.markdown(f"<div class='{box}'><div class='section-title'>{label}</div><p style='margin:.35rem 0'>Student reference: <b>{student_ref.strip()}</b></p><p style='margin:.35rem 0'>Model score: <b>{score:.1%}</b> &nbsp; | &nbsp; Review threshold: <b>50%</b></p></div>",unsafe_allow_html=True)
                st.progress(score)
                st.caption("The score is not calibrated and is not a confirmed probability. It is a prompt for staff review, not a diagnosis or an automated decision. Consider the student's circumstances and available support.")
            except Exception as exc:
                st.error(f"The model could not complete this assessment: {exc}")

with results_tab:
    st.markdown("<h2 class='section-title'>Model evaluation</h2><p class='section-help'>This page is for reviewing the experiment's performance. It is separate from student assessments.</p>",unsafe_allow_html=True)
    rows=metrics.copy().rename(columns={"stage":"Prediction point","model":"Model","threshold":"Threshold","accuracy":"Accuracy","dropout_recall":"Dropout recall","dropout_precision":"Dropout precision","roc_auc":"ROC-AUC"})
    rows["Prediction point"]=rows["Prediction point"].replace({"Stage 1":"At enrollment","Stage 2":"After first semester"})
    st.dataframe(rows[["Prediction point","Model","Threshold","Accuracy","Dropout recall","Dropout precision","ROC-AUC"]].style.format({k:"{:.3f}" for k in ["Threshold","Accuracy","Dropout recall","Dropout precision","ROC-AUC"]}),hide_index=True,use_container_width=True)
    m1,m2=st.columns(2)
    with m1: st.image(ROOT/"results"/"plots"/"stage_1_shap.png",caption="At enrollment: features the random forest relied on most")
    with m2: st.image(ROOT/"results"/"plots"/"stage_2_shap.png",caption="After first semester: features the random forest relied on most")
    st.markdown("<p class='small-copy'>Both stages use the same stratified test split. Recall measures how many students in the test set who later dropped out were flagged. SHAP shows patterns used by the model; it does not show causes. Enrolled and Graduate are grouped as non-dropout for this experiment.</p>",unsafe_allow_html=True)

st.markdown("<hr><p class='small-copy'>Research prototype · Student information entered here is not saved.</p>",unsafe_allow_html=True)
