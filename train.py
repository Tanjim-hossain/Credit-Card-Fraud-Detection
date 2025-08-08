import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional SMOTE
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except Exception:
    ImbPipeline = None
    SMOTE = None

def build_pipeline(algo: str, use_smote: bool, smote_ratio: float):
    num_features = ["Amount", "Time"]
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_features)],
        remainder="passthrough"
    )

    if algo == "logreg":
        clf = LogisticRegression(max_iter=400, solver="lbfgs", class_weight="balanced")
    elif algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            max_samples=0.5,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )
    else:
        raise ValueError("Unsupported algo. Use 'logreg' or 'rf'.")

    if use_smote:
        if ImbPipeline is None or SMOTE is None:
            raise RuntimeError("imblearn not installed; run: pip install imbalanced-learn")
        pipe = ImbPipeline([
            ("pre", pre),
            ("smote", SMOTE(sampling_strategy=smote_ratio, k_neighbors=3, random_state=42)),
            ("clf", clf),
        ])
    else:
        pipe = Pipeline([("pre", pre), ("clf", clf)])

    return pipe

def main(args):
    data_path = args.data_path
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    assert "Class" in df.columns, "Dataset must contain 'Class' column."
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = build_pipeline(args.algo, args.use_smote, args.smote_ratio)
    pipe.fit(X_train, y_train)

    # Probabilities
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba_test = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe.named_steps["clf"], "decision_function"):
        raw = pipe.named_steps["clf"].decision_function(X_test)
        from sklearn.preprocessing import MinMaxScaler
        proba_test = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).ravel()
    else:
        proba_test = pipe.predict(X_test)

    roc = float(roc_auc_score(y_test, proba_test))
    ap = float(average_precision_score(y_test, proba_test))

    # Save
    model_path = model_dir / "model.joblib"
    dump(pipe, model_path)
    meta = {
        "algo": args.algo,
        "use_smote": args.use_smote,
        "smote_ratio": args.smote_ratio if args.use_smote else 0.0,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "roc_auc_test": roc,
        "average_precision_test": ap,
        "features": list(X.columns),
        "test_size": args.test_size,
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("Saved:", model_path)
    print("Metadata:", json.dumps(meta, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--algo", default="logreg", choices=["logreg", "rf"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--use-smote", action="store_true", help="Enable SMOTE (off by default)")
    p.add_argument("--smote-ratio", type=float, default=0.05, help="Minority/majority ratio for SMOTE")
    args = p.parse_args()
    main(args)
