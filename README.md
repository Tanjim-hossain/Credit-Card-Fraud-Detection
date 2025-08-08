# Credit Card Fraud Detection 

A production-minded, end-to-end data science project to detect fraudulent credit card transactions using the popular **Kaggle Credit Card Fraud** dataset (Time, V1–V28, Amount, Class).  
Includes: exploratory data analysis (EDA), model training (Logistic Regression & Random Forest), threshold tuning, evaluation (ROC-AUC & PR-AUC), and a minimal Streamlit app for batch scoring.

## Features
- Clean, reproducible **Jupyter Notebook** for EDA → `notebooks/credit_card_fraud_detection.ipynb`
- **Imbalanced data handling** with SMOTE (train-only)
- **Pipelines** with `ColumnTransformer` and `StandardScaler` for `Amount`/`Time`
- Models: **Logistic Regression**, **Random Forest** (XGBoost optional if available)
- **Metrics**: ROC-AUC, PR-AUC, confusion matrix, precision/recall at custom thresholds
- **Model saving** with `joblib`
- **Streamlit app** to score uploaded CSVs and flag likely frauds

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train a model
Point `--data-path` to your dataset. Your path: `"/Users/tanjimhossain/Documents/creditcard.csv"`.
If you're running from this project folder, the uploaded dataset is also available at: `"/mnt/data/creditcard.csv"`.

```bash
python train.py --data-path "/Users/tanjimhossain/Documents/creditcard.csv" --model-dir models --algo logreg
# or
python train.py --data-path "/mnt/data/creditcard.csv" --model-dir models --algo rf
```

This will create a pipeline artifact at `models/model.joblib` (and `models/metadata.json`).

### 3) Run the Streamlit app
```bash
streamlit run app.py
```
- Upload a CSV with the same columns as the training data (no `Class` required for scoring).
- Adjust the **threshold** slider to trade off precision vs. recall.
- See flagged transactions and download predictions as CSV.

### 4) Explore the Notebook
Open:
```
notebooks/credit_card_fraud_detection.ipynb
```
Walks through EDA, preprocessing, training, evaluation, and threshold tuning.

## Project Structure
```
cc-fraud-detection/
├── app.py                         # Streamlit batch-scoring app
├── requirements.txt
├── train.py                       # Train and save the model pipeline
├── README.md
├── models/
│   ├── model.joblib              # Saved after training
│   └── metadata.json             # Training metadata (algo, timestamp, scores)
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
└── src/
    └── utils.py                  # Helper functions (thresholding, plots, metrics)
```
## Results
- Read from models/metadata.json after training:

Algorithm: Logistic Regression (class-weighted)

Rows × Cols: 284,807 × 31

Test split: 20%

ROC-AUC (test): 0.972

PR-AUC (test): 0.716

At threshold = 0.50 (example run in app on full CSV):
TN=284,246, FP=69, FN=18, TP=474 → Precision=0.873, Recall=0.963, F1=0.916
## Notes
- The V1–V28 features are PCA components from the original dataset; we **only scale** `Amount` and `Time`.
- **SMOTE** is applied **only on the training set** within the pipeline, to avoid leakage.
- For real deployments, consider **calibration**, drift monitoring, and adding business rules on top of the ML score.

## License
MIT
