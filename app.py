import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Responsible AI | Research Dashboard', page_icon='📊', layout='wide')
BG='#f5f7fb'; PANEL='#ffffff'; TEAL='#087f78'; BLUE='#345bc4'; WHITE='#17263c'; MUTED='#52647b'
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1400px}
.hero{background:#eaf1fb;color:#17263c;border:1px solid #cbd9eb;border-radius:18px;padding:30px;margin-bottom:20px}
.hero .eyebrow{color:#087f78;font-size:.75rem;letter-spacing:.12em;font-weight:800}
.hero .headline{color:#17263c!important;font-size:2.4rem;font-weight:800;line-height:1.15;margin:12px 0}
.hero .sub{color:#40536b;line-height:1.6}
.hero .status{color:#087f78;font-size:.8rem;margin-top:14px}
.kpi{background:#fff;color:#17263c;border:1px solid #d5deea;border-radius:14px;padding:20px;min-height:132px;margin-bottom:14px}
.kpi .label{color:#52647b;font-size:.8rem;font-weight:650}
.kpi .value{color:#17263c!important;font-size:2.1rem;font-weight:800;margin:8px 0}
.kpi .note{color:#087f78;font-size:.8rem}
.section{font-size:1.2rem;font-weight:750;margin:22px 0 12px}
.notice p,.notice b{color:#173b42!important}
.notice{background:#eaf4f3;color:#173b42;border-left:4px solid #087f78;padding:16px 19px;border-radius:0 12px 12px 0;line-height:1.65;margin:15px 0}
[data-testid="stPlotlyChart"]{border:1px solid #d5deea;border-radius:14px;padding:10px;background:#fff}
</style>""",unsafe_allow_html=True)

def card(label,value,note):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>',unsafe_allow_html=True)

def plot(fig,height=320):
    fig.update_layout(height=height,paper_bgcolor=PANEL,plot_bgcolor=PANEL,font=dict(color=WHITE,size=13),margin=dict(l=28,r=20,t=45,b=35),legend=dict(orientation='h',y=1.15,x=0,font=dict(color=MUTED)),hoverlabel=dict(bgcolor='#ffffff',font_color=WHITE))
    fig.update_xaxes(showgrid=False,tickfont=dict(color=MUTED),linecolor='#cbd5e1')
    fig.update_yaxes(gridcolor='#e2e8f0',tickfont=dict(color=MUTED),zeroline=False)
    fig.update_layout(template='plotly_white', paper_bgcolor=PANEL, plot_bgcolor=PANEL, hovermode='closest', margin=dict(l=35,r=45,t=65,b=55), bargap=.35, bargroupgap=.12)
    fig.update_xaxes(automargin=True,title_font=dict(color=WHITE),tickfont=dict(color=MUTED))
    fig.update_yaxes(automargin=True,title_font=dict(color=WHITE),tickfont=dict(color=MUTED))
    for trace in fig.data:
        if trace.type == 'bar':
            horizontal = trace.orientation == 'h'
            vals = trace.x if horizontal else trace.y
            if trace.text is None:
                trace.text = [f'{v:+.1f} pp' if horizontal else f'{v:.1f}%' for v in vals]
            trace.update(textposition='outside', textfont=dict(color=WHITE,size=14),cliponaxis=False,marker_line_width=1,marker_line_color='#cbd5e1')
    st.plotly_chart(fig,theme=None,use_container_width=True,config={'displayModeBar':False})

@st.cache_resource(show_spinner='Training educational models on the original German Credit dataset…')
def train_simulator():
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    data = fetch_openml(data_id=31, as_frame=True)
    X = data.data.copy()
    y = (data.target.astype(str) == 'bad').astype(int)
    if len(X) != 1000 or y.nunique() != 2:
        raise ValueError('Unexpected dataset format')
    train, test, yt, yv = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    numeric = list(X.select_dtypes(include='number').columns)
    categorical = [c for c in X.columns if c not in numeric]
    models = {}
    for name, estimator in [('Logistic Regression', LogisticRegression(max_iter=2000)), ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2))]:
        prep = ColumnTransformer([('numeric', StandardScaler(), numeric), ('category', OneHotEncoder(handle_unknown='ignore'), categorical)])
        models[name] = make_pipeline(prep, estimator).fit(train, yt)
    return models, test.reset_index(drop=True)

def simulator():
    st.subheader('Would changing one detail change the AI prediction?')
    st.caption('Educational simulation only · edited historical scenario · not a lending decision tool')
    st.write('Compare two machine learning models, then change only the applicant age to investigate model sensitivity.')
    try:
        models, reference = train_simulator()
    except Exception as exc:
        st.error('The public dataset could not be loaded or the models could not be trained. Try again later. The original research pages remain available.')
        st.caption(str(exc))
        return
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown('### 01 / Build a scenario')
        # Fixed, reproducible historical baseline; all remaining inputs are disclosed.
        profile = 0
        row = reference.iloc[[profile]].copy()
        st.caption('Start with one fixed historical applicant. Edit the three inputs below; the other 17 attributes remain unchanged.')
        age = st.slider('Applicant age (years)', 19, 75, int(row.iloc[0]['age']))
        amount = st.slider('Requested credit (historical Deutsche Mark)', 250, 20000, int(row.iloc[0]['credit_amount']))
        duration = st.slider('Repayment period (months)', 4, 72, int(row.iloc[0]['duration']))
        row.loc[:, 'age'] = age
        row.loc[:, 'credit_amount'] = amount
        row.loc[:, 'duration'] = duration
        with st.expander('View the 17 fixed applicant attributes'):
            st.dataframe(row.drop(columns=['age','credit_amount','duration']).T.rename(columns={profile:'Value'}).astype(str), use_container_width=True)
    with right:
        st.markdown('### 02 / Compare model predictions')
        scores = {name: float(model.predict_proba(row)[0,1]) for name, model in models.items()}
        gauges = st.columns(2)
        for col, (name, score), color in zip(gauges, scores.items(), [TEAL, BLUE]):
            with col:
                fig = go.Figure(go.Indicator(mode='gauge+number',value=score*100,number=dict(suffix='%',font=dict(size=32,color=WHITE)),title=dict(text=name,font=dict(size=13,color=WHITE)),gauge=dict(axis=dict(range=[0,100],ticksuffix='%'),bar=dict(color=color),bgcolor='#e2e8f0',borderwidth=0)))
                plot(fig,240)
        st.caption('Estimated high-risk probability · newly trained educational models')
        gap = abs(scores['Logistic Regression']-scores['Random Forest'])
        st.write(f'Model probability difference: **{gap*100:.1f} percentage points**.')
    st.divider()
    st.markdown('### 03 / Age-only fairness experiment')
    other_age = st.slider('Comparison age (all other inputs unchanged)', 19, 75, 45)
    alternative = row.copy()
    alternative.loc[:, 'age'] = other_age
    fig = go.Figure()
    for (name, model), color in zip(models.items(), [TEAL, BLUE]):
        p = float(model.predict_proba(alternative)[0,1])
        fig.add_bar(name=name, x=[f'Original: {age} years', f'Comparison: {other_age} years'], y=[scores[name]*100,p*100], marker_color=color, text=[f'{scores[name]:.1%}',f'{p:.1%}'], textposition='auto')
        st.write(f'**{name}:** {(p-scores[name])*100:+.1f} percentage points when age changes.')
    fig.update_layout(title='Same applicant, different age',barmode='group', yaxis=dict(range=[0,110],ticksuffix='%',title='Estimated high-risk probability'))
    plot(fig)
    st.markdown('### 04 / Explore what changes the prediction')
    st.caption('One-feature sensitivity, not SHAP or causal attribution. Each comparison changes only the indicated input.')
    model_name = st.selectbox('Model to inspect', list(models))
    model = models[model_name]
    changes = []
    for feature, label, value in [('age','Age: +10 years',min(75,age+10)),('credit_amount','Amount: +1,000 DM',min(20000,amount+1000)),('duration','Duration: +12 months',min(72,duration+12))]:
        scenario = row.copy()
        scenario.loc[:,feature] = value
        changes.append((f'{feature.replace(chr(95), chr(32)).title()}: {row.iloc[0][feature]} → {value}', (float(model.predict_proba(scenario)[0,1])-scores[model_name])*100))
    fig = go.Figure(go.Bar(y=[x[0] for x in changes],x=[x[1] for x in changes],orientation='h',marker_color=[TEAL if x[1]<0 else BLUE for x in changes]))
    extent = max(1, max(abs(v) for _,v in changes))*1.45
    fig.update_layout(title='Which input shifts the risk estimate?',xaxis=dict(title='Change in estimated risk (percentage points)',range=[-extent,extent],showgrid=True,gridcolor='#e2e8f0'),yaxis=dict(autorange='reversed'))
    fig.add_vline(x=0,line_color='#64748b',line_width=1)
    plot(fig,340)
    label, delta = max(changes,key=lambda item:abs(item[1]))
    direction = 'increases' if delta > 0 else 'decreases' if delta < 0 else 'does not change'
    st.markdown(f'<div class="notice"><b>Largest change in this scenario</b><br>{label} {direction} estimated risk by {abs(delta):.1f} percentage points. Teal indicates a decrease; blue indicates an increase or no change. This is a local model sensitivity result, not a causal effect.</div>',unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>Simulation methodology & limitations</b><br>These models are newly fitted using OpenML German Credit data (ID 31), an 80/20 stratified split, seed 42, and all 20 features. They are not the original notebook models. Probabilities are not validated for real lending. Age sensitivity does not establish discrimination; edited profiles may be unrealistic.</div>',unsafe_allow_html=True)


with st.sidebar:
    st.markdown('**◈ RESEARCH LAB**')
    st.markdown('### Credit Risk & Fairness')
    st.caption('INTERACTIVE RESEARCH EXPLORER')
    st.divider()
    page=st.radio('Explore research',['Credit simulator','Overview','Model performance','Fairness assessment','Methods & limitations'])
    st.divider()
    st.markdown('[Original UCI dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)')
    st.markdown('[Research repository](https://github.com/munurusnigdha/responsible-ai-credit-fairness)')

st.markdown('''<div class="hero"><div class="eyebrow">DATA SCIENCE / RESPONSIBLE AI / RESEARCH</div><div class="headline">Responsible AI in<br>Financial Decision-Making</div><div class="sub">Explore credit-risk prediction, model performance and age-group fairness using the UCI German Credit dataset.</div><div class="status">Exploratory research</div></div>''',unsafe_allow_html=True)

if page=='Credit simulator':
    simulator()
elif page=='Overview':
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
        fig=go.Figure(go.Indicator(mode='gauge+number',value=80.4048,number={'suffix':'%','font':{'color':WHITE,'size':40}},title={'text':'BEST ROC AUC','font':{'color':MUTED}},gauge={'axis':{'range':[0,100],'tickcolor':MUTED},'bar':{'color':TEAL},'bgcolor':PANEL,'steps':[{'range':[0,60],'color':'#e2e8f0'},{'range':[60,100],'color':'#cce9e5'}]}))
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
