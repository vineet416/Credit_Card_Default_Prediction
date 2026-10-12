# Credit Card Default Prediction

An end-to-end machine learning project that predicts whether a credit card client will default on next month's payment. It covers EDA, preprocessing, model comparison, a modular training pipeline (MongoDB → transform → train) and a Streamlit app for single-client predictions.

## Project Files

| Path | Description |
|---|---|
| `notebooks/1-exploratory_data_analysis-EDA.ipynb` | Data checks, distributions, outliers, feature and target relationships |
| `notebooks/2-data_preprocessing.ipynb` | Cleaning, feature engineering, split, scaling/encoding, SMOTE |
| `notebooks/3-model_training_and_evaluation.ipynb` | 7 models tuned with CV, 4 evaluated on the test set, feature importance |
| `3-model_training_and_evaluation.ipynb` (root) | Variant of notebook 3 that adds a baseline Logistic Regression |
| `notebooks/datasets/UCI_Credit_Card.csv` | Raw dataset (30,000 rows × 25 columns) |
| `notebooks/csv_outputs/`, `*_performance_outputs/`, `visualization_outputs/` | Saved metrics tables, model plots and EDA charts |
| `src/components/` | Data ingestion (MongoDB), data transformation, model trainer |
| `src/pipeline/` | `train_pipeline.py` (training) and `predict_pipeline.py` (inference) |
| `config/model.yaml` | Hyperparameter grids for XGBoost, Gradient Boosting, KNN, Random Forest |
| `artifacts/` | `credit_data.csv`, `train.csv`, `test.csv`, `preprocessor.pkl`, `model.pkl` |
| `streamlit_app.py` | Web UI for entering client details and getting a default probability |
| `upload_data.py` | Loads the raw CSV into MongoDB |
| `*.png` (root) | Architecture diagram, code-flow diagram, app screenshots |

## Dataset

UCI "Default of Credit Card Clients" (Taiwan, April–September 2005). 30,000 clients, 24 input columns plus the target.

| Column | Description |
|---|---|
| `ID` | Client ID (dropped before modeling) |
| `LIMIT_BAL` | Credit limit (NTD) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographics (coded integers; age in years) |
| `PAY_0`, `PAY_2`–`PAY_6` | Repayment status, Sept → Apr (-2/-1/0 = no use/paid/revolving; 1+ = months late) |
| `BILL_AMT1`–`BILL_AMT6` | Monthly bill statements, Sept → Apr |
| `PAY_AMT1`–`PAY_AMT6` | Monthly payments made, Sept → Apr |
| `default.payment.next.month` | Target: 1 = default, 0 = no default |

## Approach

1. **Understand the data:** 30,000 rows, no nulls, unique IDs, 22.1% positive class.
2. **Clean:** `MARRIAGE = 0` → missing (54 rows); `EDUCATION` 0/4/5/6 → "others" (468 rows); duplicates removed (29,875 rows left).
3. **Engineer features:** age groups, average bill, average payment, average delay score, average utilization (bill ÷ limit).
4. **Prepare:** stratified 75/25 split (22,406 / 7,469), median/mode imputation, scaling, one-hot and ordinal encoding, SMOTE on train only.
5. **Model:** grid search (5-fold, ROC AUC) over Logistic Regression, Decision Tree, KNN, Random Forest, AdaBoost, Gradient Boosting, XGBoost.
6. **Evaluate and deploy:** 4 best candidates scored on the held-out test set; Random Forest chosen and served through Streamlit.

## Key Insights

Computed from the raw CSV (30,000 rows) and the saved test results.

| Finding | Number |
|---|---|
| Overall default rate | 22.1% (6,636 clients) |
| Latest repayment 2+ months late vs on time/paid (`PAY_0`) | 69.6% (3,130 clients) vs 13.8% (23,182) |
| Default rate by credit-limit quartile (lowest → highest) | 31.8% → 24.7% → 17.4% → 14.0% |
| Default rate by education: graduate / university / high school | 19.2% / 23.7% / 25.2% |
| Default rate by age: 30–34 (lowest) vs 20–24 | 19.3% vs 27.2% |
| Median average monthly payment, defaulters vs non-defaulters | 1,612 vs 2,754 NTD |
| Random Forest on test set (AUC / precision / recall / F1 / accuracy) | 0.777 / 0.570 / 0.483 / 0.523 / 0.805 |
| Test-set AUC: Gradient Boosting / XGBoost / KNN | 0.759 / 0.758 / 0.693 |
| Baseline Logistic Regression (raw features, no SMOTE) | F1 0.294, recall 0.189 |

- Payment history dominates: `PAY_0` (0.117) and `Avg_Delay_Score` (0.085) are the top Random Forest importances.
- Random Forest catches about half of defaulters (798 of 1,652) with 602 false alarms among 5,817 non-defaulters.

## Tech Stack

Python · pandas, NumPy · scikit-learn (pipelines, GridSearchCV) · imbalanced-learn (SMOTE) · XGBoost · matplotlib, seaborn · Streamlit · MongoDB (pymongo) · PyYAML · Jupyter · Dev Container (Python 3.11)

## How to Use

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app (uses the saved `artifacts/`): `streamlit run streamlit_app.py`
3. Explore: open the notebooks in order 1 → 2 → 3 and run them from the `notebooks/` folder.
4. Retrain (needs MongoDB; set `MONGO_DB_URL` in `src/constant/__init__.py`, then `python upload_data.py`): `python -m src.pipeline.train_pipeline`
5. Live demo link from the previous README: https://credit-card--default-prediction.streamlit.app/ (not re-checked).
