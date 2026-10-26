"""
Step 8 — Open-Set / Unknown Device Detection Evaluation
======================================================
Evaluates the system's ability to flag traffic as either:
  1. A Known Device (with its predicted identity)
  2. An "Unknown Device" (traffic does not sufficiently match known fingerprints)

Methodology:
- Decision rule: For input window x, anomaly score S(x) = 1.0 - max_c P(y = c | x).
- Threshold selection: S(x) is calibrated strictly on the KNOWN validation split
  using a specified percentile (e.g. 95th or 99th). No tuning on the test set.
- Leave-one-device-out evaluation: Holds out each of the 5 devices in turn to simulate
  a truly unseen transmitter, reporting:
    * Unknown Detection Rate (TPR on unseen device test windows)
    * Known False Positive Rate (FPR: fraction of known test windows misflagged as Unknown)
    * Classification accuracy on accepted/verified known test windows

Results are saved to results/openset_results.json.
"""

import os
import csv
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEED = config["dataset"]["random_seed"]
DEVICE_SUBSET = config["dataset"]["device_subset"]
N_ESTIMATORS = config["classifier"]["n_estimators"]
MAX_DEPTH = config["classifier"]["max_depth"]
MIN_SAMPLES_LEAF = config["classifier"]["min_samples_leaf"]
THRESHOLD_PERCENTILE = config.get("openset", {}).get("threshold_percentile", 95)

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_split_df(split_name):
    """Load a scaled split CSV into a DataFrame and extract feature columns."""
    filepath = os.path.join(FEATURES_DIR, f"raw_scaled_{split_name}.csv")
    df = pd.read_csv(filepath)
    meta_cols = ["device_label", "window_id", "time_block_id", "split"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return df, feature_cols


def run_leave_one_out_evaluation(train_df, val_df, test_df, feature_cols):
    """
    Run leave-one-device-out cross-validation across all candidate devices.
    Calibrates thresholds on known validation split only.
    """
    print("\n" + "=" * 90)
    print("  LEAVE-ONE-DEVICE-OUT OPEN-SET BENCHMARK")
    print(f"  Thresholds chosen on KNOWN validation split (95th & 99th percentiles)")
    print(f"  Evaluated on UNSEEN test split (no leakage)")
    print("=" * 90)

    summary_results = {}

    for held_out in DEVICE_SUBSET:
        known_devices = [d for d in DEVICE_SUBSET if d != held_out]

        # Filter train & val to known devices only
        tr = train_df[train_df["device_label"].isin(known_devices)]
        va = val_df[val_df["device_label"].isin(known_devices)]

        X_tr = tr[feature_cols].values
        y_tr = tr["device_label"].values
        X_va = va[feature_cols].values
        y_va = va["device_label"].values

        # Test partitions: known test vs unseen held-out test
        te_known = test_df[test_df["device_label"].isin(known_devices)]
        te_unknown = test_df[test_df["device_label"] == held_out]

        X_te_k = te_known[feature_cols].values
        y_te_k = te_known["device_label"].values
        X_te_u = te_unknown[feature_cols].values

        # Train model on known devices only
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH if MAX_DEPTH is not None else None,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)

        # ── 1. Threshold Calibration on Known Validation Set ────────────
        va_probs = clf.predict_proba(X_va)
        va_scores = 1.0 - np.max(va_probs, axis=1)

        th_95 = float(np.percentile(va_scores, 95))
        th_99 = float(np.percentile(va_scores, 99))

        # ── 2. Test Set Evaluation ──────────────────────────────────────
        te_k_probs = clf.predict_proba(X_te_k)
        te_k_preds = clf.predict(X_te_k)
        te_k_scores = 1.0 - np.max(te_k_probs, axis=1)

        te_u_probs = clf.predict_proba(X_te_u)
        te_u_scores = 1.0 - np.max(te_u_probs, axis=1)

        # Metrics at 95th percentile
        k_flagged_95 = te_k_scores > th_95
        u_flagged_95 = te_u_scores > th_95
        fpr_95 = float(np.mean(k_flagged_95) * 100)
        tpr_95 = float(np.mean(u_flagged_95) * 100)

        verified_mask_95 = ~k_flagged_95
        if np.sum(verified_mask_95) > 0:
            acc_verified_95 = float(accuracy_score(y_te_k[verified_mask_95], te_k_preds[verified_mask_95]) * 100)
        else:
            acc_verified_95 = 0.0

        # Metrics at 99th percentile
        k_flagged_99 = te_k_scores > th_99
        u_flagged_99 = te_u_scores > th_99
        fpr_99 = float(np.mean(k_flagged_99) * 100)
        tpr_99 = float(np.mean(u_flagged_99) * 100)

        verified_mask_99 = ~k_flagged_99
        if np.sum(verified_mask_99) > 0:
            acc_verified_99 = float(accuracy_score(y_te_k[verified_mask_99], te_k_preds[verified_mask_99]) * 100)
        else:
            acc_verified_99 = 0.0

        summary_results[held_out] = {
            "held_out_device": held_out,
            "test_unknown_windows": len(te_unknown),
            "test_known_windows": len(te_known),
            "th_95": th_95,
            "min_conf_95": float(1.0 - th_95),
            "known_fpr_95": fpr_95,
            "unknown_tpr_95": tpr_95,
            "acc_verified_95": acc_verified_95,
            "th_99": th_99,
            "min_conf_99": float(1.0 - th_99),
            "known_fpr_99": fpr_99,
            "unknown_tpr_99": tpr_99,
            "acc_verified_99": acc_verified_99,
        }

    # Print Report Table
    print(f"\n  {'Held-Out Unknown Device':<26} | {'Windows':>7} | {'Val 95%ile Thresh (5% target)':^30} | {'Val 99%ile Thresh (1% target)':^30}")
    print(f"  {'':<26} | {'':>7} | {'Known FPR':>10} {'Unknown TPR':>13} {'Acc(Ver)':>10} | {'Known FPR':>10} {'Unknown TPR':>13} {'Acc(Ver)':>10}")
    print("  " + "-" * 105)

    all_tpr_95 = []
    all_fpr_95 = []
    all_tpr_99 = []
    all_fpr_99 = []

    for d, res in summary_results.items():
        all_tpr_95.append(res["unknown_tpr_95"])
        all_fpr_95.append(res["known_fpr_95"])
        all_tpr_99.append(res["unknown_tpr_99"])
        all_fpr_99.append(res["known_fpr_99"])

        print(f"  {d:<26} | {res['test_unknown_windows']:>7} | "
              f"{res['known_fpr_95']:>9.1f}% {res['unknown_tpr_95']:>12.1f}% {res['acc_verified_95']:>9.1f}% | "
              f"{res['known_fpr_99']:>9.1f}% {res['unknown_tpr_99']:>12.1f}% {res['acc_verified_99']:>9.1f}%")

    print("  " + "-" * 105)
    print(f"  {'AVERAGE ACROSS DEVICES':<26} | {'':>7} | "
          f"{np.mean(all_fpr_95):>9.1f}% {np.mean(all_tpr_95):>12.1f}% {'—':>10} | "
          f"{np.mean(all_fpr_99):>9.1f}% {np.mean(all_tpr_99):>12.1f}% {'—':>10}")
    print("=" * 108)

    return summary_results


def evaluate_five_device_known_threshold(train_df, val_df, test_df, feature_cols):
    """
    Fit Random Forest on all 5 known devices and calibrate the operational open-set
    threshold on the full validation split.
    """
    print("\n" + "=" * 90)
    print("  FULL 5-DEVICE MODEL OPEN-SET THRESHOLD CALIBRATION")
    print("=" * 90)

    X_tr = train_df[feature_cols].values
    y_tr = train_df["device_label"].values

    X_va = val_df[feature_cols].values
    y_va = val_df["device_label"].values

    X_te = test_df[feature_cols].values
    y_te = test_df["device_label"].values

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH if MAX_DEPTH is not None else None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    # Validation scores
    va_probs = clf.predict_proba(X_va)
    va_scores = 1.0 - np.max(va_probs, axis=1)

    th_selected = float(np.percentile(va_scores, THRESHOLD_PERCENTILE))
    min_confidence = float(1.0 - th_selected)

    # Test evaluation
    te_probs = clf.predict_proba(X_te)
    te_preds = clf.predict(X_te)
    te_scores = 1.0 - np.max(te_probs, axis=1)

    te_flagged = te_scores > th_selected
    te_fpr = float(np.mean(te_flagged) * 100)

    verified_mask = ~te_flagged
    acc_all = float(accuracy_score(y_te, te_preds) * 100)
    acc_verified = float(accuracy_score(y_te[verified_mask], te_preds[verified_mask]) * 100)

    print(f"  Validation Threshold ({THRESHOLD_PERCENTILE}th percentile): S_anomaly > {th_selected:.4f}")
    print(f"  Minimum Classification Confidence Required:        {min_confidence * 100:.2f}%")
    print(f"  Known Test Windows Flagged as Unknown (FPR):       {te_fpr:.2f}% ({np.sum(te_flagged)}/{len(te_flagged)})")
    print(f"  Overall Test Accuracy (All Windows):               {acc_all:.2f}%")
    print(f"  Verified Test Accuracy (Accepted Windows):         {acc_verified:.2f}%")

    return {
        "calibrated_threshold_percentile": THRESHOLD_PERCENTILE,
        "calibrated_anomaly_threshold": th_selected,
        "minimum_confidence_required": min_confidence,
        "test_known_false_positive_rate": te_fpr,
        "test_overall_accuracy": acc_all,
        "test_verified_accuracy": acc_verified,
        "total_test_windows": len(te_flagged),
        "verified_test_windows": int(np.sum(verified_mask)),
        "flagged_test_windows": int(np.sum(te_flagged)),
    }


def main():
    print("=" * 90)
    print("  STEP 8: OPEN-SET / UNKNOWN DEVICE DETECTION EVALUATION")
    print(f"  Feature set: raw statistical (13 features)")
    print(f"  Validation calibration percentile: {THRESHOLD_PERCENTILE}th")
    print("=" * 90)

    train_df, feature_cols = load_split_df("train")
    val_df, _ = load_split_df("val")
    test_df, _ = load_split_df("test")

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} windows")

    # 1. Leave-one-out evaluation across devices
    loo_results = run_leave_one_out_evaluation(train_df, val_df, test_df, feature_cols)

    # 2. Five-device operational threshold calibration
    five_dev_results = evaluate_five_device_known_threshold(train_df, val_df, test_df, feature_cols)

    # 3. Save combined results
    out_payload = {
        "leave_one_out_benchmarks": loo_results,
        "operational_five_device_calibration": five_dev_results,
    }

    out_file = os.path.join(RESULTS_DIR, "openset_results.json")
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\n  Saved open-set benchmark results: {out_file}")
    print(f"  Step 8 complete.")


if __name__ == "__main__":
    main()
