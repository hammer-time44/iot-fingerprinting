"""
Evaluate unknown devices and known controls through the trained model.
Computes validation-calibrated threshold, unknown detection rates, and false alarm rates.
"""
import os
import pickle
import numpy as np
import pandas as pd

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

# 1. Load model and scaler
model_path = os.path.join(RESULTS_DIR, "step7_20261026_151020", "rf_model.pkl")
if not os.path.exists(model_path):
    model_path = os.path.join(RESULTS_DIR, "step7_20260917_144006", "rf_raw_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

scaler_path = os.path.join(RESULTS_DIR, "raw_stat_scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler_dict = pickle.load(f)

mean_vec = scaler_dict["mean"]
std_vec = scaler_dict["std"]
feat_cols = scaler_dict["feature_names"]

# 2. Validation threshold calculation
df_val = pd.read_csv(os.path.join(FEATURES_DIR, "raw_scaled_val.csv"))
X_val = df_val[feat_cols].values
probs_val = model.predict_proba(X_val)
confs_val = np.max(probs_val, axis=1)

# 5th percentile of validation confidence: exactly 5% of known validation windows fall below this
threshold_5th = float(np.percentile(confs_val, 5))

# 3. Known test false alarm calculation
df_test = pd.read_csv(os.path.join(FEATURES_DIR, "raw_scaled_test.csv"))
X_test = df_test[feat_cols].values
probs_test = model.predict_proba(X_test)
confs_test = np.max(probs_test, axis=1)
flagged_test_count = int(np.sum(confs_test < threshold_5th))
known_fpr = (flagged_test_count / len(confs_test)) * 100

print("=" * 80)
print(f"VALIDATION CALIBRATED THRESHOLD: {threshold_5th:.4f}")
print("Methodology: 5th percentile of max class probability across 1,071 validation windows")
print(f"Known Test False Alarm Rate (< {threshold_5th:.4f}): {flagged_test_count}/{len(confs_test)} = {known_fpr:.2f}%")
print("=" * 80)

# 4. Unknown devices evaluation
df_unk = pd.read_csv(os.path.join(FEATURES_DIR, "unknown_devices_features.csv"))

results_table = []

# Known test controls
for dev in sorted(df_test["device_label"].unique()):
    sub = df_test[df_test["device_label"] == dev]
    X_d = sub[feat_cols].values
    probs_d = model.predict_proba(X_d)
    confs_d = np.max(probs_d, axis=1)
    preds_d = model.predict(X_d)

    labels, counts = np.unique(preds_d, return_counts=True)
    dom_label = labels[np.argmax(counts)]
    dom_share = (np.max(counts) / len(preds_d)) * 100
    low_conf_share = float(np.mean(confs_d < threshold_5th) * 100)

    results_table.append({
        "Device": dev,
        "Type": "Known (Test)",
        "Windows": len(sub),
        "Dominant Prediction": f"{dom_label} ({dom_share:.1f}%)",
        "Mean Confidence": float(np.mean(confs_d)),
        "Low-Conf Flagged %": low_conf_share,
    })

# Unknown devices
total_unk_windows = 0
total_unk_flagged = 0

for dev in sorted(df_unk["device_label"].unique()):
    sub = df_unk[df_unk["device_label"] == dev]
    raw_X = sub[feat_cols].values
    scaled_X = (raw_X - mean_vec) / std_vec

    probs_d = model.predict_proba(scaled_X)
    confs_d = np.max(probs_d, axis=1)
    preds_d = model.predict(scaled_X)

    labels, counts = np.unique(preds_d, return_counts=True)
    dom_label = labels[np.argmax(counts)]
    dom_share = (np.max(counts) / len(preds_d)) * 100
    low_conf_count = int(np.sum(confs_d < threshold_5th))
    low_conf_share = float(np.mean(confs_d < threshold_5th) * 100)

    total_unk_windows += len(sub)
    total_unk_flagged += low_conf_count

    results_table.append({
        "Device": dev,
        "Type": "Unknown Device",
        "Windows": len(sub),
        "Dominant Prediction": f"{dom_label} ({dom_share:.1f}%)",
        "Mean Confidence": float(np.mean(confs_d)),
        "Low-Conf Flagged %": low_conf_share,
    })

overall_unk_flagged_rate = (total_unk_flagged / total_unk_windows) * 100

df_res = pd.DataFrame(results_table)
res_path = os.path.join(RESULTS_DIR, "unknown_evaluation_table.csv")
df_res.to_csv(res_path, index=False)

print("\nEVALUATION SUMMARY TABLE (ALL DEVICES):")
print(df_res.to_string(index=False))

print("\n" + "=" * 80)
print(f"OVERALL UNKNOWN DETECTION RATE: {total_unk_flagged}/{total_unk_windows} windows ({overall_unk_flagged_rate:.2f}%)")
print(f"KNOWN TEST FALSE ALARM RATE:    {flagged_test_count}/{len(confs_test)} windows ({known_fpr:.2f}%)")
print("=" * 80)
