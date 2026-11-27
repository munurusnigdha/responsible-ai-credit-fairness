# US Mortgage Lending & Responsible AI

Exploratory research on historical mortgage lending decisions in Massachusetts using 2023 Home Mortgage Disclosure Act (HMDA) data. This is not a default prediction or lending decision system.

## Results

Of 210,643 reported records, 149,649 applications had action codes 1, 2 or 3. A stratified 80/20 split yielded 29,930 test cases.

| Model | Accuracy | ROC-AUC | Denial recall | F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 77.35% | .500 | 0% | .000 |
| Logistic Regression | 77.42% | .724 | 1.22% | .024 |
| Random Forest | 80.24% | .782 | 20.22% | .317 |
| Gradient Boosting | 81.37% | .798 | 28.44% | .409 |

Gradient Boosting performed best among the tested configurations, but missed most denials at threshold 0.5. Demographic attributes were excluded from training and retained for descriptive fairness audits. Group disparities do not establish discrimination.

## Reproduce

Download the [2023 Massachusetts HMDA CSV](https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=MA&years=2023) as `state_MA.csv`.

```bash
pip install -r requirements.txt
python hmda_analysis.py state_MA.csv
streamlit run app.py
```

The dashboard reads the saved `hmda_results.json`; the raw CSV is not required for deployment. The analysis uses eight financial and mortgage features, training-only preprocessing, and random seed 42. Runtime measurements are machine-specific.

## Limitations

Single held-out split, no external validation, no threshold optimization, possible decision-time feature availability issues, no causal or legal conclusions, and no measured organizational cost savings. Fairness groups below 100 test records are omitted.

[Interactive dashboard](https://responsible-ai-credit-fairness-gxlp4v9gk8ayiqrtddbvge.streamlit.app/)

The earlier German Credit notebook and HTML dashboard are retained as a separate historical benchmark study.
