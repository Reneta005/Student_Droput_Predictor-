"""Comparable two-stage student dropout evaluation with optional SHAP plots."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
S1=['Admission grade','Previous qualification (grade)','Age at enrollment','Course','Application mode','Daytime/evening attendance','Scholarship holder','Debtor','Gender','Education Special Needs']
S2=S1+['Curricular units 1st sem (approved)','Curricular units 1st sem (grade)']
CATS=['Course','Application mode','Daytime/evening attendance','Scholarship holder','Debtor','Gender','Education Special Needs']
def prep(cols):
    cat=[c for c in CATS if c in cols]; num=[c for c in cols if c not in cat]
    return ColumnTransformer([('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat),('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num)])
def main():
    p=argparse.ArgumentParser(); p.add_argument('data',type=Path); p.add_argument('--output-dir',type=Path,default=Path('results')); p.add_argument('--threshold',type=float,default=.5); a=p.parse_args()
    out=a.output_dir; (out/'models').mkdir(parents=True,exist_ok=True); (out/'plots').mkdir(parents=True,exist_ok=True); d=pd.read_csv(a.data); y=(d.Target=='Dropout').astype(int)
    tr,te=train_test_split(np.arange(len(d)),test_size=.2,random_state=42,stratify=y); yt,ye=y.iloc[tr],y.iloc[te]
    methods=[('Logistic regression',LogisticRegression(class_weight='balanced',max_iter=3000,solver='liblinear',random_state=42)),('Random forest',RandomForestClassifier(n_estimators=500,class_weight='balanced_subsample',min_samples_leaf=2,n_jobs=-1,random_state=42))]
    rows=[]; shap_models={}
    for stage,cols in [('Stage 1',S1),('Stage 2',S2)]:
        for name,est in methods:
            model=Pipeline([('preprocess',prep(cols)),('model',est)]); model.fit(d.iloc[tr][cols],yt); prob=model.predict_proba(d.iloc[te][cols])[:,1]; pred=(prob>=a.threshold).astype(int)
            rows.append({'stage':stage,'model':name,'threshold':a.threshold,'accuracy':accuracy_score(ye,pred),'dropout_recall':recall_score(ye,pred,zero_division=0),'dropout_precision':precision_score(ye,pred,zero_division=0),'roc_auc':roc_auc_score(ye,prob)})
            dump(model,out/'models'/f"{stage.lower().replace(' ','_')}_{name.lower().replace(' ','_')}.joblib")
            if name=='Random forest': shap_models[stage]=(model,d.iloc[te][cols].iloc[:120])
    m=pd.DataFrame(rows); m.to_csv(out/'metrics_comparison.csv',index=False); (out/'metrics_comparison.md').write_text('| stage | model | threshold | accuracy | dropout recall | dropout precision | ROC-AUC |\n|---|---|---:|---:|---:|---:|---:|\n' + ''.join(f'| {r.stage} | {r.model} | {r.threshold:.2f} | {r.accuracy:.3f} | {r.dropout_recall:.3f} | {r.dropout_precision:.3f} | {r.roc_auc:.3f} |\n' for r in m.itertuples(index=False)), encoding='utf-8')
    for col in ['accuracy','dropout_recall','dropout_precision','roc_auc']:
        q=m.pivot(index='model',columns='stage',values=col); q['Stage 2 minus Stage 1']=q['Stage 2']-q['Stage 1']; q.to_csv(out/f'{col}_stage_change.csv')
    try:
        import shap
        for stage,(model,sample) in shap_models.items():
            trf=model.named_steps['preprocess'].transform(sample); trf=trf.toarray() if hasattr(trf,'toarray') else np.asarray(trf,dtype=float); names=model.named_steps['preprocess'].get_feature_names_out(); vals=shap.TreeExplainer(model.named_steps['model']).shap_values(trf)
            if isinstance(vals,list): vals=vals[1]
            elif getattr(vals,'ndim',0)==3: vals=vals[:,:,1]
            importance=np.abs(vals).mean(axis=0); order=np.argsort(importance)[-12:]; fig,ax=plt.subplots(figsize=(9,6)); ax.barh(np.asarray(names)[order],importance[order],color='#2186d4'); ax.set_xlabel('Mean |SHAP value| (impact on dropout score)'); ax.set_title(f'{stage}: Random forest SHAP importance'); fig.tight_layout(); fig.savefig(out/'plots'/f"{stage.lower().replace(' ','_')}_shap.png",dpi=180,bbox_inches='tight'); plt.close(fig)
    except ImportError: (out/'SHAP_NOT_INSTALLED.txt').write_text('Install requirements.txt and rerun train_evaluate.py to generate genuine SHAP plots. No substitute explanations were fabricated.\n',encoding='utf-8')
    (out/'split.json').write_text(json.dumps({'random_state':42,'test_size':.2,'stratified':True,'train_rows':len(tr),'test_rows':len(te),'threshold':a.threshold},indent=2),encoding='utf-8'); print(m.to_string(index=False))
if __name__=='__main__': main()
