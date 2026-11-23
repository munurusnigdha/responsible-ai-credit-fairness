import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Responsible AI | Research Dashboard', page_icon='📊', layout='wide')
BG='#07111f'; PANEL='#11243a'; TEAL='#24d2c1'; BLUE='#6798ff'; WHITE='#ecf6ff'; MUTED='#b5cbdc'
st.markdown('''<style>
:root{color-scheme:dark}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#07111f!important;color:#ecf6ff!important}
[data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:#102036!important;border-right:1px solid #31516b!important}
[data-testid="stHeader"]{background:transparent!important}
.block-container{padding-top:2rem!important;max-width:1400px!important}
.stApp p,.stApp span,.stApp label,.stApp h1,.stApp h2,.stApp h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label{color:#ecf6ff!important}
[data-testid="stSidebar"] a,.stApp a{color:#6ce6dc!important}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:9px 12px;border-radius:9px}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#1b3b54}
[data-testid="stSelectbox"] div[data-baseweb="select"]>div{background:#17324b!important;color:#ecf6ff!important}
[data-testid="stPlotlyChart"]{background:#11243a;border:1px solid #2a4a65;border-radius:17px;padding:10px}
.hero{background:linear-gradient(110deg,#183a55,#0b192b 75%);border:1px solid #31516b;border-radius:20px;padding:32px;margin-bottom:22px}
.hero .eyebrow{color:#57e6d9!important;font-size:.75rem;letter-spacing:.15em;font-weight:800}
.hero .headline{color:#f0f8ff!important;font-size:2.6rem;font-weight:800;line-height:1.14;letter-spacing:-.04em;margin:12px 0}
.hero .sub{color:#c0d5e4!important;max-width:800px;line-height:1.6}
.hero .status{display:inline-block;color:#91f2e9!important;background:#17414a;border:1px solid #33727b;border-radius:30px;padding:7px 13px;margin-top:15px;font-size:.78rem}
.kpi{background:linear-gradient(140deg,#17324b,#11243a);border:1px solid #31516b;border-radius:16px;padding:20px 22px;min-height:132px;margin-bottom:14px}
.kpi .label{color:#b5cbdc!important;font-size:.8rem;font-weight:650}
.kpi .value{color:#f0f8ff!important;font-size:2.2rem;font-weight:800;margin:8px 0 3px}
.kpi .note{color:#64e3d8!important;font-size:.76rem}
.section{font-size:1.2rem;color:#f0f8ff!important;font-weight:750;margin:22px 0 12px}
.notice{background:#142d41;border-left:4px solid #24d2c1;padding:16px 19px;border-radius:0 12px 12px 0;color:#e2f2ff!important;line-height:1.65;margin:15px 0}
.notice *{color:#e2f2ff!important}
</style>''',unsafe_allow_html=True)

def card(label,value,note):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>',unsafe_allow_html=True)

def plot(fig,height=320):
    fig.update_layout(height=height,paper_bgcolor=PANEL,plot_bgcolor=PANEL,font=dict(color=WHITE,size=13),margin=dict(l=28,r=20,t=45,b=35),legend=dict(orientation='h',y=1.15,x=0,font=dict(color=MUTED)),hoverlabel=dict(bgcolor='#17324b',font_color=WHITE))
    fig.update_xaxes(showgrid=False,tickfont=dict(color=MUTED),linecolor='#31516b')
    fig.update_yaxes(gridcolor='#28445b',tickfont=dict(color=MUTED),zeroline=False)
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

with st.sidebar:
    st.markdown('**◈ RESEARCH LAB**')
    st.markdown('### Credit Risk & Fairness')
    st.caption('INTERACTIVE RESEARCH EXPLORER')
    st.divider()
    page=st.radio('Explore research',['Overview','Model performance','Fairness assessment','Methods & limitations'])
    st.divider()
    st.markdown('[Original UCI dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)')
    st.markdown('[Research repository](https://github.com/munurusnigdha/responsible-ai-credit-fairness)')

st.markdown('''<div class="hero"><div class="eyebrow">DATA SCIENCE / RESPONSIBLE AI / RESEARCH</div><div class="headline">Responsible AI in<br>Financial Decision-Making</div><div class="sub">Explore credit-risk prediction, model performance and age-group fairness using the UCI German Credit dataset.</div><div class="status">● Published research summary</div></div>''',unsafe_allow_html=True)

if page=='Overview':
    cols=st.columns(4)
    for col,items in zip(cols,[('DATASET RECORDS','1,000','20 features'),('TEST CASES','200','Held-out evaluation'),('BEST ACCURACY','78%','Logistic Regression'),('BEST ROC AUC','0.804','Logistic Regression')]):
        with col:card(*items)
    st.markdown('<div class="section">Model performance</div>',unsafe_allow_html=True)
    a,b=st.columns([1.5,1])
    with a:
        fig=go.Figure()
        fig.add_bar(name='Accuracy',x=['Logistic Regression','Random Forest'],y=[78,76],marker_color=TEAL,text=['78%','76%'],textposition='outside')
        fig.add_bar(name='ROC AUC',x=['Logistic Regression','Random Forest'],y=[80.4048,78.375],marker_color=BLUE,text=['80.4%','78.4%'],textposition='outside')
        fig.update_layout(barmode='group',yaxis=dict(range=[0,100],ticksuffix='%'))
        plot(fig)
    with b:
        fig=go.Figure(go.Indicator(mode='gauge+number',value=80.4048,number={'suffix':'%','font':{'color':WHITE,'size':40}},title={'text':'BEST ROC AUC','font':{'color':MUTED}},gauge={'axis':{'range':[0,100],'tickcolor':MUTED},'bar':{'color':TEAL},'bgcolor':PANEL,'steps':[{'range':[0,60],'color':'#1d364b'},{'range':[60,100],'color':'#20515c'}]}))
        plot(fig)
    st.markdown('<div class="section">Fairness snapshot</div>',unsafe_allow_html=True)
    a,b=st.columns([1.5,1])
    with a:
        fig=go.Figure()
        x=['Observed high risk','LR predicted high risk','RF predicted high risk']
        fig.add_bar(name='Under 35',x=x,y=[39,31.4,26.7],marker_color=TEAL)
        fig.add_bar(name='Age 35+',x=x,y=[20,15.8,10.5],marker_color=BLUE)
        fig.update_layout(barmode='group',yaxis=dict(range=[0,48],ticksuffix='%'))
        plot(fig,310)
    with b:
        st.markdown('<div class="notice"><b>Interpret with care</b><p>Group differences are descriptive, not evidence of causal discrimination.</p><p>Under 35: 105 test cases<br>Age 35+: 95 test cases</p></div>',unsafe_allow_html=True)
        st.write('Use the sidebar to explore model comparisons and false-positive rates.')
    st.caption('Previously calculated research results. This is not a live lending system.')
elif page=='Model performance':
    st.markdown('<div class="section">Interactive model comparison</div>',unsafe_allow_html=True)
    model=st.selectbox('Select model',['Logistic Regression','Random Forest'])
    acc,auc=(78,80.4048) if model=='Logistic Regression' else (76,78.375)
    a,b,c=st.columns(3)
    with a:card('ACCURACY',f'{acc}%','Held-out test set')
    with b:card('ROC AUC',f'{auc/100:.3f}','Ranking ability')
    with c:card('TEST CASES','200','Same evaluation split')
    fig=go.Figure(go.Bar(x=['Accuracy','ROC AUC'],y=[acc,auc],marker_color=[TEAL,BLUE],text=[f'{acc}%',f'{auc:.1f}%'],textposition='outside'))
    fig.update_layout(yaxis=dict(range=[0,100],ticksuffix='%'))
    plot(fig,380)
    st.markdown('<div class="notice">An ROC curve requires case-level prediction scores; no curve has been fabricated from summary results.</div>',unsafe_allow_html=True)
elif page=='Fairness assessment':
    st.markdown('<div class="section">Age-group fairness explorer</div>',unsafe_allow_html=True)
    model=st.selectbox('Select model',['Logistic Regression','Random Forest'])
    metric=st.radio('Choose metric',['Predicted high risk','False positive rate'],horizontal=True)
    data={'Logistic Regression':{'Predicted high risk':[31.4,15.8],'False positive rate':[14.1,9.2]},'Random Forest':{'Predicted high risk':[26.7,10.5],'False positive rate':[14.1,5.3]}}
    values=data[model][metric]
    a,b,c=st.columns(3)
    with a:card('UNDER 35',f'{values[0]:.1f}%','n = 105')
    with b:card('AGE 35+',f'{values[1]:.1f}%','n = 95')
    with c:card('GROUP GAP',f'{values[0]-values[1]:.1f} pp','Descriptive difference')
    fig=go.Figure(go.Bar(x=['Under 35 (n=105)','Age 35+ (n=95)'],y=values,marker_color=[TEAL,BLUE],text=[f'{v:.1f}%' for v in values],textposition='outside',width=.5))
    fig.update_layout(yaxis=dict(range=[0,max(40,max(values)*1.3)],ticksuffix='%'))
    plot(fig,380)
    st.markdown('<div class="notice">Observed high-risk rates were 39% (under 35) and 20% (35+). This exploratory comparison does not establish discrimination.</div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="section">Study design and limitations</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a:card('RECORDS','1,000','UCI Statlog German Credit')
    with b:card('FEATURES','20','Original dataset')
    with c:card('TEST CASES','200','Held-out evaluation')
    st.subheader('Methodology')
    st.write('Logistic Regression and Random Forest were evaluated using the completed Python/Colab research. This dashboard presents the existing results without retraining.')
    st.subheader('Interpretation limits')
    st.markdown('<div class="notice">Fairness comparisons use an exploratory under-35 versus 35+ split (105 and 95 test cases). Small samples and historical data limit generalizability. Group differences are not causal proof of discrimination.</div>',unsafe_allow_html=True)
    st.write('This is not a live lending system and does not make individual credit decisions.')
