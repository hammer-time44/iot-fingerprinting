"""
Step 5 — Raw Statistical Feature Extraction
============================================
For each windowed log1p-IAT file from Step 4:

  Compute 13 raw statistical features per window:
    mean, std, min, max, median, skew, kurtosis,
    P25, P75, P95, P99, energy, entropy

  Output saved to data/features/raw_stat_features.csv with
  device_label and time_block_id carried through.

  No scaling or normalisation — that happens in Step 6,
  fit on the training split only.
"""

import os
import csv
import yaml
import numpy as np

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE_SUBSET = config["dataset"]["device_subset"]
WINDOW_SIZE = config["preprocessing"]["window_size"]

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")


# ══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def shannon_entropy(values):
    """Compute Shannon entropy using normalised squared magnitudes."""
    sq = np.array(values) ** 2
    total = np.sum(sq)
    if total == 0:
        return 0.0
    probs = sq / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_raw_stats(window_values):
    """
    Compute 13 raw statistical features directly on the
    log1p-transformed IAT window.
    Returns a dict of feature_name -> float.
    """
    arr = np.array(window_values, dtype=np.float64)

    mean_val = float(np.mean(arr))
    std_val  = float(np.std(arr))
    min_val  = float(np.min(arr))
    max_val  = float(np.max(arr))
    median_val = float(np.median(arr))

    # Skewness (Fisher's definition)
    if std_val > 0:
        skew_val = float(np.mean(((arr - mean_val) / std_val) ** 3))
    else:
        skew_val = 0.0

    # Excess kurtosis (Fisher's definition)
    if std_val > 0:
        kurt_val = float(np.mean(((arr - mean_val) / std_val) ** 4) - 3.0)
    else:
        kurt_val = 0.0

    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    energy  = float(np.sum(arr ** 2))
    entropy = shannon_entropy(arr)

    return {
        "raw_mean":     mean_val,
        "raw_std":      std_val,
        "raw_min":      min_val,
        "raw_max":      max_val,
        "raw_median":   median_val,
        "raw_skew":     skew_val,
        "raw_kurtosis": kurt_val,
        "raw_p25":      p25,
        "raw_p75":      p75,
        "raw_p95":      p95,
        "raw_p99":      p99,
        "raw_energy":   energy,
        "raw_entropy":  entropy,
    }


RAW_FEATURE_NAMES = [
    "raw_mean", "raw_std", "raw_min", "raw_max", "raw_median",
    "raw_skew", "raw_kurtosis", "raw_p25", "raw_p75", "raw_p95", "raw_p99",
    "raw_energy", "raw_entropy",
]


# ══════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════

def load_windows(device_name):
    """Load windowed data from Step 4 output."""
    filepath = os.path.join(FEATURES_DIR, f"{device_name}_windows.csv")
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return []

    windows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iat_values = [float(row[f"iat_{i}"]) for i in range(WINDOW_SIZE)]
            windows.append({
                "device_label":  row["device_label"],
                "window_id":     int(row["window_id"]),
                "time_block_id": int(row["time_block_id"]),
                "iat_values":    iat_values,
            })
    return windows


def process_device(device_name):
    """Extract raw statistical features for every window of a device."""
    windows = load_windows(device_name)
    if not windows:
        return []

    rows = []
    for w in windows:
        meta = {
            "device_label":  w["device_label"],
            "window_id":     w["window_id"],
            "time_block_id": w["time_block_id"],
        }
        feats = compute_raw_stats(w["iat_values"])
        rows.append({**meta, **feats})
    return rows


def save_features(rows, filename):
    """Save feature rows to CSV."""
    if not rows:
        return None

    filepath = os.path.join(FEATURES_DIR, filename)
    meta_cols = ["device_label", "window_id", "time_block_id"]
    all_cols  = meta_cols + RAW_FEATURE_NAMES

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for k, v in row.items():
                formatted[k] = f"{v:.9f}" if isinstance(v, float) else v
            writer.writerow(formatted)
    return filepath


def main():
    print("=" * 80)
    print("  STEP 5: RAW STATISTICAL FEATURE EXTRACTION")
    print(f"  Window size: {WINDOW_SIZE} | Features per window: {len(RAW_FEATURE_NAMES)}")
    print("=" * 80)

    all_rows = []
    device_counts = {}

    for device_name in DEVICE_SUBSET:
        print(f"\n  Processing: {device_name}")
        rows = process_device(device_name)
        print(f"    {len(rows):,} windows extracted")
        device_counts[device_name] = len(rows)
        all_rows.extend(rows)

    # ── Save combined feature file ─────────────────────────────────────
    out_path = save_features(all_rows, "raw_stat_features.csv")
    print(f"\n  Saved: {out_path}")
    print(f"  Total rows: {len(all_rows):,}  |  Columns: {3 + len(RAW_FEATURE_NAMES)} "
          f"(3 meta + {len(RAW_FEATURE_NAMES)} features)")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n  {'Device':<28} {'Windows':>10}")
    print(f"  {'-'*40}")
    for device_name in DEVICE_SUBSET:
        print(f"  {device_name:<28} {device_counts.get(device_name, 0):>10,}")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<28} {len(all_rows):>10,}")

    print(f"\n  Feature columns ({len(RAW_FEATURE_NAMES)}):")
    for i, name in enumerate(RAW_FEATURE_NAMES):
        print(f"    {i:>2}. {name}")

    print(f"\n  Step 5 complete.")


if __name__ == "__main__":
    main()
