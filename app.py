import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Responsible AI | Credit Risk & Fairness', page_icon='📊', layout='wide')
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#091321 !important;color:#e9f4ff !important;}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {background:#101e30 !important;color:#e9f4ff !important;}
.stApp p, .stApp span, .stApp label, .stApp small, .stApp div[data-testid="stCaptionContainer"],
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color:#e9f4ff !important;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {color:#eaf8ff !important;}
[data-testid="stMetric"] {background:#12243a !important;border:1px solid #2a4d68 !important;padding:18px !important;border-radius:15px !important;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {color:#b9d4e7 !important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color:#f1f8ff !important;}
[data-testid="stSidebar"] a {color:#66d9e8 !important;}
[data-testid="stSidebar"] [role="radiogroup"] label {background:transparent !important;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#1c344e !important;border-radius:8px;}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {background:#152b43 !important;color:#eaf8ff !important;border-color:#315573 !important;}
.research-notice {background:#142e43 !important;color:#e8f5ff !important;border:1px solid #315a70 !important;border-left:4px solid #21c6bb !important;border-radius:10px !important;padding:17px 20px !important;margin:14px 0 22px !important;font-size:1rem !important;line-height:1.55 !important;}
[data-testid="stAlert"] {background:#142e43 !important;color:#e8f5ff !important;}
[data-testid="stAlert"] * {color:#e8f5ff !important;}
[data-testid="stHeader"] {background:#091321 !important;}
</style>
""", unsafe_allow_html=True)
st.title('Responsible AI in Financial Decision-Making')
st.caption('German Credit dataset · Exploratory credit-risk prediction and age-group fairness assessment · 1,000 records')
st.markdown('''<div class="research-notice">Published research summary based on a held-out test set of 200 cases. This is not a live lending or decision-making system.</div>''', unsafe_allow_html=True)

page = st.sidebar.radio('Explore research', ['Overview', 'Model performance', 'Fairness assessment', 'Methods & limitations'])
st.sidebar.markdown('[Original UCI dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)')
st.sidebar.markdown('[Research repository](https://github.com/munurusnigdha/responsible-ai-credit-fairness)')

if page == 'Overview':
    a,b,c,d=st.columns(4)
    a.metric('Records','1,000');b.metric('Test cases','200');c.metric('Best accuracy','78%');d.metric('Best ROC AUC','0.804')
    st.subheader('Model comparison')
    df=pd.DataFrame({'Model':['Logistic Regression','Random Forest'],'Accuracy':[.78,.76],'ROC AUC':[.804048,.783750]})
    fig=px.bar(df,x='Model',y=['Accuracy','ROC AUC'],barmode='group',color_discrete_sequence=['#21c6bb','#548cf5'],range_y=[0,1],template='plotly_dark')
    fig.update_layout(paper_bgcolor='#091321',plot_bgcolor='#12243a',legend_title_text='',yaxis_tickformat='.0%',font=dict(color='#e9f4ff'),xaxis=dict(tickfont=dict(color='#c9dced'),title_font=dict(color='#e9f4ff')),yaxis=dict(tickfont=dict(color='#c9dced'),gridcolor='#35516b'),legend=dict(font=dict(color='#e9f4ff')))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Values are from the completed Colab analysis, not recomputed in this app.')
elif page == 'Model performance':
    model=st.selectbox('Select model',['Logistic Regression','Random Forest'])
    acc,auc=(.78,.804048) if model=='Logistic Regression' else (.76,.783750)
    x,y=st.columns(2);x.metric('Test accuracy',f'{acc:.0%}');y.metric('ROC AUC',f'{auc:.3f}')
    fig=go.Figure(go.Indicator(mode='gauge+number',value=auc*100,title={'text':'ROC AUC (out of 100)'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#21c6bb'},'bgcolor':'#12243a'}))
    fig.update_layout(paper_bgcolor='#091321',font_color='#e9f4ff',font=dict(color='#e9f4ff'),height=330)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('An ROC curve requires case-level prediction scores; no curve is fabricated from summary metrics.')
elif page == 'Fairness assessment':
    model=st.selectbox('Select model',['Logistic Regression','Random Forest'])
    metrics={'Logistic Regression':{'Predicted high risk':[31.4,15.8],'False positive rate':[14.1,9.2]},'Random Forest':{'Predicted high risk':[26.7,10.5],'False positive rate':[14.1,5.3]}}
    measure=st.radio('Metric',['Predicted high risk','False positive rate'],horizontal=True)
    df=pd.DataFrame({'Age group':['Under 35 (n=105)','35+ (n=95)'],'Percent':metrics[model][measure]})
    fig=px.bar(df,x='Age group',y='Percent',color='Age group',text='Percent',color_discrete_sequence=['#21c6bb','#548cf5'],template='plotly_dark')
    fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
    fig.update_layout(showlegend=False,paper_bgcolor='#091321',plot_bgcolor='#12243a',font=dict(color='#e9f4ff'),yaxis_title='Percent',yaxis_range=[0,45],xaxis=dict(tickfont=dict(color='#c9dced')),yaxis=dict(tickfont=dict(color='#c9dced'),gridcolor='#35516b'))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Observed high-risk rate: under 35 = 39%; age 35+ = 20%. Age split is exploratory. Group differences alone do not establish discrimination.')
else:
    st.subheader('Study design')
    st.write('UCI Statlog German Credit dataset: 1,000 records, 20 features. Logistic Regression and Random Forest were evaluated on a held-out test set of 200 cases.')
    st.subheader('Interpretation limits')
    st.write('Fairness comparisons use an exploratory under-35 versus 35+ split (105 and 95 test cases). These are descriptive results from a small test set, not causal evidence of discrimination. Historical credit data may contain biases and may not represent present-day applicants.')
    st.write('This dashboard presents previously calculated results. It does not retrain models, provide individual credit decisions, or simulate new outcomes.')
