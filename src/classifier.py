"""
Step 7 — Random Forest Classifier
==================================
1. Load scaled raw statistical features from Step 6.
2. Train a Random Forest on the training split.
3. Tune against validation split only; touch test split exactly once.
4. Report: accuracy, macro F1, per-device precision/recall/F1,
   and a confusion matrix saved to the timestamped results folder.
5. Save model, config snapshot, and results summary.
"""

import os
import csv
import pickle
import json
import datetime
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEED             = config["dataset"]["random_seed"]
DEVICE_SUBSET    = config["dataset"]["device_subset"]
N_ESTIMATORS     = config["classifier"]["n_estimators"]
MAX_DEPTH        = config["classifier"]["max_depth"]
MIN_SAMPLES_LEAF = config["classifier"]["min_samples_leaf"]
THRESHOLD_PERCENTILE = config.get("openset", {}).get("threshold_percentile", 95)

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR  = os.path.join(PROJ_ROOT, "results")

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR   = os.path.join(RESULTS_DIR, f"step7_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)


def load_split(split_name):
    """Load a scaled feature split CSV; return X (features) and y (labels)."""
    filepath = os.path.join(FEATURES_DIR, f"raw_scaled_{split_name}.csv")
    X, y, meta = [], [], []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in reader.fieldnames
                        if c not in ("device_label", "window_id",
                                     "time_block_id", "split")]
        for row in reader:
            X.append([float(row[c]) for c in feature_cols])
            y.append(row["device_label"])
            meta.append({"window_id": row["window_id"],
                          "time_block_id": row["time_block_id"]})
    return np.array(X), np.array(y), feature_cols, meta


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Random Forest and evaluate on validation then test."""
    print(f"\n  Training Random Forest...")
    print(f"    n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}, seed={SEED}")
    print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH if MAX_DEPTH is not None else None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Validation ─────────────────────────────────────────────────────
    y_val_pred    = clf.predict(X_val)
    val_acc       = accuracy_score(y_val, y_val_pred)
    val_f1_macro  = f1_score(y_val, y_val_pred, average="macro")

    print(f"\n  --- Validation ---")
    print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"  Macro F1: {val_f1_macro:.4f}")
    print(f"\n  Classification Report (Val):")
    print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))

    val_cm = confusion_matrix(y_val, y_val_pred, labels=DEVICE_SUBSET)

    # ── Test (touch exactly once) ───────────────────────────────────────
    y_test_pred   = clf.predict(X_test)
    test_acc      = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

    print(f"\n  --- Test ---")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Macro F1: {test_f1_macro:.4f}")
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    test_cm = confusion_matrix(y_test, y_test_pred, labels=DEVICE_SUBSET)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, labels=DEVICE_SUBSET, zero_division=0
    )
    per_class = {}
    for i, device in enumerate(DEVICE_SUBSET):
        per_class[device] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }

    return clf, {
        "val_accuracy":  val_acc,
        "val_f1_macro":  val_f1_macro,
        "val_cm":        val_cm,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
        "test_cm":       test_cm,
        "per_class":     per_class,
    }


def predict_with_rejection(clf, X, threshold):
    """
    Predict known device label or 'Unknown Device' if anomaly score exceeds threshold.
    Anomaly score S(x) = 1.0 - max_c P(y = c | x).
    Threshold is calibrated on the validation split.
    """
    probs = clf.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    anomaly_scores = 1.0 - max_probs
    base_preds = clf.predict(X)

    output = []
    for pred, score in zip(base_preds, anomaly_scores):
        if score > threshold:
            output.append("Unknown Device")
        else:
            output.append(pred)
    return np.array(output), anomaly_scores


def plot_confusion_matrix(cm, labels, title, filepath):
    """Save confusion matrix as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    short_labels = [l[:12] for l in labels]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=short_labels,
           yticklabels=short_labels,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    print("=" * 80)
    print("  STEP 7: RANDOM FOREST CLASSIFIER")
    print(f"  Features: raw statistical (13 per window)")
    print(f"  Config: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}")
    print(f"  Results dir: {RUN_DIR}")
    print("=" * 80)

    # ── Load splits ────────────────────────────────────────────────────
    print("\n  Loading scaled features...")
    X_train, y_train, feature_cols, _ = load_split("train")
    X_val,   y_val,   _,            _ = load_split("val")
    X_test,  y_test,  _,            _ = load_split("test")

    print(f"  Features: {len(feature_cols)} cols")
    print(f"  Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # ── Train & Evaluate ───────────────────────────────────────────────
    clf, results = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # ── Feature importances ────────────────────────────────────────────
    print(f"\n  Feature Importances (top {len(feature_cols)}):")
    importances = clf.feature_importances_
    for rank, idx in enumerate(np.argsort(importances)[::-1]):
        print(f"    {rank+1:>2}. {feature_cols[idx]:<28} {importances[idx]:.4f}")

    # ── TPLink warning (small test set) ───────────────────────────────
    tp_support = results["per_class"]["TPLinkSmartPlug"]["support"]
    if tp_support < 20:
        print(f"\n  WARNING: TPLinkSmartPlug test set has only {tp_support} samples.")
        print(f"    Per-class metrics for this device are statistically unstable.")

    # ── Confusion matrices ─────────────────────────────────────────────
    print(f"\n  Saving confusion matrices...")
    plot_confusion_matrix(
        results["test_cm"], DEVICE_SUBSET,
        "Random Forest — Test Confusion Matrix",
        os.path.join(RUN_DIR, "cm_test.png"),
    )
    plot_confusion_matrix(
        results["val_cm"], DEVICE_SUBSET,
        "Random Forest — Validation Confusion Matrix",
        os.path.join(RUN_DIR, "cm_val.png"),
    )

    # ── Save model ─────────────────────────────────────────────────────
    model_path = os.path.join(RUN_DIR, "rf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Model saved: {model_path}")

    # ── Config snapshot ────────────────────────────────────────────────
    config_path = os.path.join(RUN_DIR, "config_snapshot.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Open-Set / Unknown Device Threshold Calibration ──────────────
    print(f"\n{'='*70}")
    print(f"  OPEN-SET THRESHOLD CALIBRATION (Validation {THRESHOLD_PERCENTILE}th percentile)")
    print(f"{'='*70}")
    val_probs = clf.predict_proba(X_val)
    val_anomaly_scores = 1.0 - np.max(val_probs, axis=1)
    openset_threshold = float(np.percentile(val_anomaly_scores, THRESHOLD_PERCENTILE))
    min_confidence = float(1.0 - openset_threshold)

    # Evaluate rejection on known test split
    test_preds_openset, test_anomaly_scores = predict_with_rejection(clf, X_test, openset_threshold)
    test_flagged = (test_preds_openset == "Unknown Device")
    known_fpr = float(np.mean(test_flagged) * 100)
    print(f"  Calibrated Anomaly Threshold: S(x) > {openset_threshold:.4f} (min confidence: {min_confidence*100:.2f}%)")
    print(f"  Known Test Windows Flagged as Unknown (FPR): {known_fpr:.2f}% ({np.sum(test_flagged)}/{len(X_test)})")

    # ── Save Results Summary ───────────────────────────────────────────
    summary = {
        "timestamp":       TIMESTAMP,
        "val_accuracy":    results["val_accuracy"],
        "val_f1_macro":    results["val_f1_macro"],
        "test_accuracy":   results["test_accuracy"],
        "test_f1_macro":   results["test_f1_macro"],
        "per_class":       results["per_class"],
        "openset": {
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "anomaly_threshold": openset_threshold,
            "min_confidence_required": min_confidence,
            "known_test_fpr": known_fpr,
        },
        "config": {
            "n_estimators":     N_ESTIMATORS,
            "max_depth":        MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "seed":             SEED,
        },
    }
    summary_path = os.path.join(RUN_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results summary: {summary_path}")

    print(f"\n  Step 7 complete.")


if __name__ == "__main__":
    main()
