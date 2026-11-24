import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title='US Mortgage Lending | Responsible AI', layout='wide')
URL='https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=MA&years=2023'
COLS=['action_taken','loan_amount','income','property_value','applicant_age','derived_race','derived_sex','loan_purpose','occupancy_type','lien_status']
FEATURES=['loan_amount','income','property_value']
@st.cache_data(show_spinner=False)
def load(source):
    d=pd.read_csv(source,usecols=COLS,low_memory=False)
    for c in FEATURES+['action_taken','loan_purpose','occupancy_type','lien_status']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[d.action_taken.isin([1,2,3]) & d.loan_purpose.eq(1) & d.occupancy_type.eq(1) & d.lien_status.eq(1)].copy()
    d['denied']=d.action_taken.eq(3).astype(int)
    for c in FEATURES: d.loc[d[c]<=0,c]=float('nan')
    return d
@st.cache_resource(show_spinner=False)
def train(d):
    a,b=train_test_split(d,test_size=.2,random_state=42,stratify=d.denied)
    models={'Logistic regression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000)), 'Random forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=100,min_samples_leaf=20,random_state=42,n_jobs=-1))}
    for m in models.values(): m.fit(a[FEATURES],a.denied)
    return models,b
st.title('US Mortgage Lending & Algorithmic Fairness')
st.write('Can machine learning predict mortgage denials, and how do its errors differ across demographic groups?')
st.caption('2023 Massachusetts HMDA public loan-level data • Home purchases • Owner-occupied • First lien')
with st.sidebar:
    st.header('Research workspace')
    page=st.radio('Explore',['Lending landscape','Model comparison','Fairness audit','What-if explorer','Methodology'])
    st.link_button('Official HMDA data', 'https://ffiec.cfpb.gov/data-browser/')
    uploaded=st.file_uploader('Optional: upload Massachusetts HMDA CSV',type=['csv'])
try:
    with st.spinner('Loading public mortgage records…'):
        d=load(uploaded if uploaded is not None else URL)
    if d.denied.nunique()!=2 or len(d)<100: raise ValueError('Insufficient eligible records or outcome classes.')
except Exception as e:
    st.error('Unable to load eligible HMDA records. Download the 2023 Massachusetts CSV from the official data browser and upload it in the sidebar.')
    st.caption(str(e)); st.stop()
if uploaded is not None: st.warning('Using uploaded data. Verify that it contains 2023 Massachusetts records; geography and year are not independently validated.')
def money(x): return f'${x:,.0f}'
def chart(fig):
    fig.update_layout(template='plotly_white',font=dict(color='#172b42'),paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',margin=dict(l=20,r=20,t=35,b=35))
    st.plotly_chart(fig,use_container_width=True,theme=None)
if page=='Lending landscape':
    a,b,c=st.columns(3); a.metric('Decided applications',f'{len(d):,}'); b.metric('Denial rate',f'{d.denied.mean():.1%}'); c.metric('Median loan amount',money(d.loan_amount.median()))
    st.subheader('Where do lending outcomes differ?')
    group=st.selectbox('Compare by',['applicant_age','derived_race','derived_sex'],format_func=lambda x:{'applicant_age':'Age','derived_race':'Race','derived_sex':'Sex'}[x])
    g=d.groupby(group).denied.agg(['mean','size']).reset_index(); g=g[g['size']>=30]; g['Denial rate (%)']=100*g['mean']; g['Applications']=g['size']
    chart(px.bar(g.sort_values('Denial rate (%)'),x='Denial rate (%)',y=group,orientation='h',text_auto='.1f',hover_data=['Applications'],color_discrete_sequence=['#176b87']))
    st.info('These are observed lending decisions, not borrower default rates. Differences do not establish discrimination.')
elif page=='Methodology':
    st.markdown('**Population:** Action codes 1 (originated), 2 (approved but not accepted), 3 (denied). Home purchase, owner occupied, first lien. Other outcomes excluded.\n\n**Models:** Logistic regression and random forest, stratified 80/20 split, seed 42. Median imputation fitted on training data. Predictors: loan amount, reported annual income (thousands of dollars), property value. Demographic variables excluded from predictors and retained for auditing.\n\n**Limitations:** Historical lender decisions are not ground truth for creditworthiness. Limited predictors, omitted confounders, possible proxy effects and no causal or legal fairness conclusions. This is not a lending tool.')
else:
    with st.spinner('Training models on the selected HMDA data…'): models,test=train(d)
    if page=='Model comparison':
        rows=[]
        for name,m in models.items():
            p=m.predict_proba(test[FEATURES])[:,1]; rows.append({'Model':name,'Accuracy':accuracy_score(test.denied,p>=.5),'ROC AUC':roc_auc_score(test.denied,p)})
        st.subheader('Held-out model performance'); st.dataframe(pd.DataFrame(rows).style.format({'Accuracy':'{:.1%}','ROC AUC':'{:.3f}'}),hide_index=True)
        st.caption(f'{len(test):,} held-out applications. Accuracy can be misleading when denials are uncommon.')
    elif page=='Fairness audit':
        name=st.selectbox('Model',list(models)); group=st.selectbox('Demographic group',['applicant_age','derived_race','derived_sex']); threshold=st.slider('Denial classification threshold',.05,.95,.5,.05)
        t=test.copy(); t['prediction']=models[name].predict_proba(t[FEATURES])[:,1]>=threshold
        rows=[]
        for key,g in t.groupby(group):
            approved=g[g.denied==0]; denied=g[g.denied==1]
            if len(g)>=30 and len(approved)>=10 and len(denied)>=10: rows.append({'Group':str(key),'Applications':len(g),'Observed denial (%)':100*g.denied.mean(),'Predicted denial (%)':100*g.prediction.mean(),'False positive (%)':100*approved.prediction.mean(),'False negative (%)':100*(1-denied.prediction.mean())})
        if rows:
            table=pd.DataFrame(rows); metric=st.selectbox('Compare metric',list(table.columns[2:])); chart(px.bar(table,x=metric,y='Group',orientation='h',text_auto='.1f',hover_data=['Applications'],color_discrete_sequence=['#176b87'])); st.dataframe(table.round(2),hide_index=True)
        else: st.warning('No groups meet minimum sample requirements.')
        st.info('False positives are approved applications predicted as denied. Group differences are descriptive, not proof of discrimination.')
    else:
        st.subheader('Explore how financial inputs affect predicted denial')
        a,b,c=st.columns(3); loan=a.number_input('Loan amount ($)',1000,5000000,350000,5000); income=b.number_input('Annual income ($)',1000,2000000,100000,5000); value=c.number_input('Property value ($)',1000,10000000,450000,5000)
        x=pd.DataFrame([[loan,income/1000,value]],columns=FEATURES)
        for col,(name,m) in zip(st.columns(2),models.items()): col.metric(name,f'{m.predict_proba(x)[0,1]:.1%}',help='Estimated probability of historical denial label, not default risk.')
        st.info('Educational model only. Not an approval recommendation or an estimate of your personal eligibility. Age and other demographic attributes are not model inputs.')
