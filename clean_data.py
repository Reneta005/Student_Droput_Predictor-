"""Clean and validate the supplied semicolon-separated dataset."""
import argparse, csv, json
from pathlib import Path
import pandas as pd

def load_and_clean(source):
    with Path(source).open('r', encoding='utf-8-sig', newline='') as f:
        line=f.readline().strip().lstrip('\ufeff')
        if line.startswith('"') and line.endswith('"'): line=line[1:-1].replace('""','"')
        headers=[x.strip().strip('"').strip() for x in next(csv.reader([line], delimiter=';'))]
        df=pd.read_csv(f, sep=';', header=None, names=headers)
    df=df.rename(columns={'Nacionality':'Nationality','Educational special needs':'Education Special Needs'})
    report={'rows':int(len(df)), 'columns_including_target':int(len(df.columns)), 'duplicate_rows':int(df.duplicated().sum()), 'missing_cells':int(df.isna().sum().sum()), 'target_counts':df.Target.value_counts(dropna=False).to_dict()}
    if len(df.columns)!=37 or 'Target' not in df: raise ValueError(f'Expected 36 inputs + Target; got {len(df.columns)}')
    if df.Target.isna().any() or not set(df.Target.unique()) <= {'Dropout','Enrolled','Graduate'}: raise ValueError('Missing or unexpected target labels')
    for c in ['Daytime/evening attendance','Scholarship holder','Debtor','Gender','Education Special Needs']:
        if not set(df[c].dropna().unique()) <= {0,1}: raise ValueError(f'Invalid binary values in {c}')
    report['unique_values']={c:sorted(df[c].dropna().unique().tolist()) for c in ['Application mode','Course','Previous qualification','Daytime/evening attendance','Scholarship holder','Debtor','Gender','Education Special Needs']}
    return df,report

def main():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--output',type=Path,default=Path('cleaned_data.csv')); p.add_argument('--report',type=Path,default=Path('data_quality.json')); a=p.parse_args()
    df,r=load_and_clean(a.input); a.output.parent.mkdir(parents=True,exist_ok=True); a.report.parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.output,index=False); a.report.write_text(json.dumps(r,indent=2,default=str),encoding='utf-8'); print(json.dumps(r,indent=2,default=str))
if __name__=='__main__': main()
