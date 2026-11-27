import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Mortgage Lending | Responsible AI', layout='wide')
r = json.loads(Path(__file__).with_name('hmda_results.json').read_text())
m = pd.DataFrame(r['models']).set_index('model')
f = pd.DataFrame(r['fairness'])
best = m.loc['Gradient Boosting']
base = m.loc['Majority baseline']
race = f[(f.model == 'Gradient Boosting') & (f.attribute == 'derived_race')].set_index('group')
b = race.loc['Black or African American']; w = race.loc['White']
st.title('Can AI learn mortgage decisions fairly?')
st.caption('Independent exploratory research | Massachusetts, 2023 | Home Mortgage Disclosure Act data')
st.markdown('A mortgage decision can determine whether a family can buy a home. This study asks whether machine learning can reproduce historical lending decisions, and whether those predictions differ across demographic groups.')
a,c,d = st.columns(3)
a.metric('Mortgage records reviewed', f"{r['total']:,}")
c.metric('Applications used in study', f"{r['eligible']:,}")
d.metric('Unseen applications tested', f"{r['test']:,}")
st.subheader('What did the research find?')
st.markdown(f'''**1. AI learned useful patterns, but accuracy is misleading.** The best model matched {best.accuracy:.1%} of historical decisions, compared with {base.accuracy:.1%} for a rule that always predicts approval.\n\n**2. Most historical denials were missed.** At the tested threshold, the best model detected only {best.recall:.1%} of actual denials.\n\n**3. Removing demographic inputs did not eliminate differences.** Historical denial rates and model-predicted denial rates differed across racial groups.''')
st.warning('This study predicts historical decisions, not repayment or creditworthiness. It does not establish discrimination or justify automated loan decisions.')
t1,t2,t3,t4 = st.tabs(['Can AI predict decisions?', 'Are outcomes different?', 'Explore a mortgage example', 'Research & business implications'])
with t1:
 st.subheader('What happens for every 100 historically denied applications?')
 q=m.reset_index(); q['Detected']=q.recall*100; q['Missed']=100-q.Detected
 fig=px.bar(q,x='model',y=['Detected','Missed'],barmode='stack',labels={'model':'','value':'Out of 100 historical denials','variable':'Outcome'},color_discrete_map={'Detected':'#167D83','Missed':'#B9C4D0'},template='plotly_white')
 st.plotly_chart(fig,use_container_width=True)
 st.write(f'Gradient Boosting detects about {best.recall*100:.0f} of every 100 historical denials and misses about {(1-best.recall)*100:.0f}. A high accuracy score alone is not enough to justify deployment.')
 st.markdown(f'**Business implication:** A lender should evaluate missed cases and incorrect flags before using a model for review prioritization. The measured accuracy gain is {(best.accuracy-base.accuracy)*100:.2f} percentage points, not a measured financial return.')
 with st.expander('Technical model comparison'):
  st.dataframe(m.style.format('{:.3f}'),use_container_width=True)
  st.caption('ROC-AUC measures ranking ability; recall measures the fraction of historical denials detected. Timings depend on hardware.')
with t2:
 st.subheader('How often were applications denied?')
 a,c=st.columns(2);a.metric('Black applicants',f'{b.observed_denial:.1%}');c.metric('White applicants',f'{w.observed_denial:.1%}')
 st.write(f'The historical denial rate for Black applicants was {b.observed_denial/w.observed_denial:.2f} times the rate for White applicants in the test sample. This is an unadjusted comparison, not evidence of a causal effect of race.')
 model=st.selectbox('Choose a model',m.index.tolist(),index=3)
 attr=st.selectbox('Compare groups by',['derived_race','derived_sex','applicant_age'],format_func=lambda x:{'derived_race':'Race','derived_sex':'Sex','applicant_age':'Age'}[x])
 z=f[(f.model==model)&(f.attribute==attr)].copy();z['group']=z.group.replace({'8888':'Not applicable','9999':'Not available'})
 q=z.rename(columns={'observed_denial':'Historical denial rate','predicted_denial':'AI-predicted denial rate'}).melt(id_vars=['group','n'],value_vars=['Historical denial rate','AI-predicted denial rate'],var_name='Measure',value_name='Rate')
 fig=px.bar(q,x='Rate',y='group',color='Measure',barmode='group',orientation='h',text=q.Rate.map(lambda v:f'{v:.1%}'),hover_data=['n'],template='plotly_white',color_discrete_sequence=['#456B94','#167D83']);fig.update_xaxes(tickformat='.0%');fig.update_layout(yaxis_title='',height=max(400,65*len(z)))
 st.plotly_chart(fig,use_container_width=True)
 st.write('A smaller predicted denial gap does not automatically mean a fairer model: it may also reflect missed denials. Differences in financial characteristics and other unmeasured factors require further investigation.')
 with st.expander('Detailed fairness audit and sample sizes'):
  st.dataframe(z.drop(columns=['model','attribute']).style.format({k:'{:.1%}' for k in ['observed_denial','predicted_denial','fpr','tpr']}),hide_index=True,use_container_width=True)
  st.caption('FPR: historically approved cases predicted denied. TPR: historical denials detected. Groups with fewer than 100 test records omitted.')
with t3:
 st.subheader('Explore how a mortgage relates to income and property value')
 st.write('Change the figures to understand two financial ratios relevant to lending analysis. This is an educational calculator, not the trained AI model.')
 income=st.slider('Annual household income ($)',25000,300000,85000,5000)
 loan=st.slider('Requested mortgage ($)',50000,1000000,300000,10000)
 value=st.slider('Property value ($)',50000,1500000,400000,10000)
 a,c=st.columns(2);a.metric('Loan / annual income',f'{loan/income:.2f}×');c.metric('Loan / property value',f'{loan/value:.1%}')
 st.write(f'The mortgage is {loan/income:.2f} times annual income and represents {loan/value:.1%} of the property value. Neither ratio alone establishes affordability or eligibility.')
 st.info('No approval probability is calculated. Interest rates, existing debts, credit history and other factors are needed for a meaningful individual assessment.')
with t4:
 st.subheader('How could a lender use this research?')
 st.markdown('**Model governance:** compare candidate models against a simple baseline and monitor errors.\n\n**Fairness oversight:** investigate demographic differences and audit errors across groups before deployment.\n\n**Human review:** retain accountable review processes rather than using this model to approve or deny applicants.\n\n**Next research step:** validate on other years and states, evaluate thresholds and investigate confounding factors. These are proposed actions, not measured business outcomes.')
 st.subheader('How the study was conducted')
 st.write(f"Of {r['total']:,} records, {r['eligible']:,} had action codes 1, 2 or 3 (originated, approved but not accepted, or denied). Other actions were excluded. A stratified 80/20 split used {r['train']:,} training cases and {r['test']:,} test cases (seed 42).")
 st.write('Eight inputs: loan amount, income, property value, loan type, loan purpose, occupancy, lien status and construction method. Median imputation and one-hot encoding were fitted on training data. Race, sex and age were excluded from model inputs and used for auditing. Classification threshold: 0.5.')
 st.write('Limitations: one state and year, one train/test split, no external validation, no causal inference, and possible post-decision information in reported attributes. No repayment outcome or financial return was measured.')
 st.download_button('Download research results',json.dumps(r,indent=2),'hmda_results.json','application/json')
