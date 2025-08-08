import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, classification_report

def pick_first_existing(paths):
    """Return the first existing path from a list; otherwise return the first element as default."""
    for p in paths:
        try:
            if p and isinstance(p, str) and len(p) > 0 and (p == "-" or p == "/dev/stdin"):
                # allow stdin-ish, but not used here
                pass
        except Exception:
            pass
        try:
            import os
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return paths[0] if paths else None

def evaluate_at_threshold(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    return {
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": precision, "recall": recall, "f1": f1
    }

def plot_roc_pr(y_true, proba):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

def best_threshold_by_f1(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    thresholds = np.append(thresholds, 1.0)  # align sizes
    f1 = (2*precision*recall)/(precision+recall+1e-12)
    idx = np.nanargmax(f1)
    return thresholds[idx], f1[idx]
