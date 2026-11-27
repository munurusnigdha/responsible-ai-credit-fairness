import pandas as pd,numpy as np,time,json,sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
num=['loan_amount','income','property_value'];cat=['loan_type','loan_purpose','occupancy_type','lien_status','construction_method'];groups=['derived_race','derived_sex','applicant_age']
d=pd.read_csv(sys.argv[1],usecols=num+cat+groups+['action_taken','activity_year','state_code'],low_memory=False);total=len(d);counts=d.action_taken.value_counts().to_dict();d=d[d.action_taken.isin([1,2,3])].copy();y=(d.action_taken==3).astype(int);X=d[num+cat].copy()
for c in num:X[c]=pd.to_numeric(X[c],errors='coerce')
for c in cat:X[c]=X[c].astype(str)
a,b=train_test_split(np.arange(len(d)),test_size=.2,random_state=42,stratify=y)
prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
models={'Majority baseline':DummyClassifier(strategy='most_frequent'),'Logistic Regression':LogisticRegression(max_iter=500),'Random Forest':RandomForestClassifier(n_estimators=40,max_depth=10,min_samples_leaf=10,n_jobs=4,random_state=42),'Gradient Boosting':HistGradientBoostingClassifier(max_iter=100,random_state=42)}
results=[];audits=[]
for name,model in models.items():
 m=make_pipeline(prep,model);t=time.perf_counter();m.fit(X.iloc[a],y.iloc[a]);fit=time.perf_counter()-t;t=time.perf_counter();prob=m.predict_proba(X.iloc[b])[:,1];pred=(prob>=.5).astype(int);elapsed=time.perf_counter()-t;yt=y.iloc[b].to_numpy();r=dict(model=name,accuracy=accuracy_score(yt,pred),balanced_accuracy=balanced_accuracy_score(yt,pred),precision=precision_score(yt,pred,zero_division=0),recall=recall_score(yt,pred),f1=f1_score(yt,pred),auc=roc_auc_score(yt,prob),fit_seconds=fit,predict_seconds=elapsed);print(r,flush=True);results.append(r)
 for g in groups:
  for label,idx in d.iloc[b].reset_index(drop=True).groupby(g).indices.items():
   if len(idx)<100:continue
   yy=yt[idx];pp=pred[idx];audits.append(dict(model=name,attribute=g,group=str(label),n=len(idx),observed_denial=float(yy.mean()),predicted_denial=float(pp.mean()),fpr=float(pp[yy==0].mean()) if (yy==0).any() else None,tpr=float(pp[yy==1].mean()) if (yy==1).any() else None))
out={'total':total,'action_counts':counts,'eligible':len(d),'train':len(a),'test':len(b),'denial_rate':float(y.mean()),'missing':X.isna().sum().to_dict(),'models':results,'fairness':audits};Path(__file__).with_name('hmda_results.json').write_text(json.dumps(out,indent=2));print('DONE',flush=True)
