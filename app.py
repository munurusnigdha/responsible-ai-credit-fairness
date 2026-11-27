import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path
st.set_page_config(page_title='US Mortgage Lending | Responsible AI',layout='wide')
r=json.loads(Path(__file__).with_name('hmda_results.json').read_text())
st.title('US Mortgage Lending & Responsible AI')
st.caption('Massachusetts • 2023 HMDA • Historical decision prediction, not default prediction')
a,b,c=st.columns(3);a.metric('Applications analyzed',f"{r['eligible']:,}");b.metric('Held-out test cases',f"{r['test']:,}");c.metric('Historical denial rate',f"{r['denial_rate']:.1%}")
m=pd.DataFrame(r['models']);f=pd.DataFrame(r['fairness'])
t1,t2,t3=st.tabs(['Model comparison','Fairness audit','Research methods'])
with t1:
 st.subheader('Does accuracy hide missed denials?')
 metric=st.selectbox('Compare models by',['recall','accuracy','auc','f1','balanced_accuracy','precision','fit_seconds','predict_seconds'])
 fig=px.bar(m,x='model',y=metric,text_auto='.3f',color='model',template='plotly_white');fig.update_layout(showlegend=False,xaxis_title='',yaxis_title=metric.replace('_',' ').title());st.plotly_chart(fig,use_container_width=True)
 st.dataframe(m,hide_index=True,use_container_width=True)
 st.info('Gradient Boosting ranks highest on ROC-AUC and denial recall among these configurations, but misses most denials at the default threshold.')
with t2:
 model=st.selectbox('Model',m.model.tolist(),index=3);attribute=st.selectbox('Demographic attribute',['derived_race','derived_sex','applicant_age'],format_func=lambda x:{'derived_race':'Race','derived_sex':'Sex','applicant_age':'Age'}[x]);z=f[(f.model==model)&(f.attribute==attribute)].copy();z['group']=z.group.replace({'8888':'Not applicable','9999':'Not available'});metrics=st.multiselect('Metrics',['observed_denial','predicted_denial','fpr','tpr'],default=['observed_denial','predicted_denial']);
 if metrics:
  q=z.melt(id_vars=['group','n'],value_vars=metrics,var_name='metric',value_name='rate');fig=px.bar(q,x='rate',y='group',color='metric',barmode='group',orientation='h',text=q.rate.map(lambda v:f'{v:.1%}'),hover_data=['n'],template='plotly_white');fig.update_xaxes(tickformat='.0%');st.plotly_chart(fig,use_container_width=True)
 st.dataframe(z.drop(columns=['model','attribute']),hide_index=True,use_container_width=True);st.caption('Groups with fewer than 100 test records omitted. FPR: approved cases predicted denied; TPR: denied cases correctly identified. Disparities are descriptive, not proof of discrimination.')
with t3:
 st.write('2023 Massachusetts HMDA. Action codes 1 and 2 are positive lending outcomes; code 3 is denial. Other actions excluded. Stratified 80/20 split, seed 42. Eight features: loan amount, income, property value, loan type, loan purpose, occupancy, lien status and construction method. Median imputation and one-hot encoding fitted on training data. Demographics excluded from model inputs. Threshold 0.5.')
 st.warning('Single split; no external validation or causal inference. Historical reported financial attributes may not all be available before a decision. Not suitable for individual lending decisions. Timings are environment-specific.')
 st.download_button('Download measured research results',json.dumps(r,indent=2),'hmda_results.json','application/json')
