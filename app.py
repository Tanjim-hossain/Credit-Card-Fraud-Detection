import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from src.utils import evaluate_at_threshold

st.set_page_config(page_title="Credit Card Fraud Scorer", layout="wide")

st.title("💳 Credit Card Fraud Scorer")
st.write("Upload transactions, choose a threshold, and review flagged records.")

model_path = Path("models/model.joblib")
if not model_path.exists():
    st.warning("No trained model found at `models/model.joblib`. Train one with `python train.py --data-path <path>` first.")
else:
    model = load(model_path)

uploaded = st.file_uploader("Upload CSV (same columns as training data; `Class` optional)", type=["csv"])
threshold = st.slider("Decision threshold (higher → fewer flags, higher precision)", 0.0, 1.0, 0.50, 0.01)

if uploaded is not None and model_path.exists():
    df = pd.read_csv(uploaded)
    y_true = None
    if "Class" in df.columns:
        y_true = df["Class"].astype(int).values
        X = df.drop(columns=["Class"])
    else:
        X = df.copy()

    # Predict probabilities
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        if hasattr(model.named_steps["clf"], "decision_function"):
            raw = model.named_steps["clf"].decision_function(X)
            from sklearn.preprocessing import MinMaxScaler
            mms = MinMaxScaler()
            proba = mms.fit_transform(raw.reshape(-1,1)).ravel()
        else:
            proba = model.predict(X)

    preds = (proba >= threshold).astype(int)
    out = df.copy()
    out["fraud_score"] = proba
    out["is_fraud_pred"] = preds

    st.subheader("Flagged Transactions")
    flagged = out[out["is_fraud_pred"] == 1]
    st.write(f"Total flagged: **{flagged.shape[0]}** / {out.shape[0]}")
    st.dataframe(flagged.head(100))

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    if y_true is not None:
        metrics = evaluate_at_threshold(y_true, proba, threshold=threshold)
        st.subheader("Validation (if ground-truth `Class` provided)")
        st.json(metrics)
