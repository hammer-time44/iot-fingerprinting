"""
Step 6 — Leakage-Safe Train/Val/Test Split + Scaling
=====================================================
1. Load raw statistical feature set from Step 5.
2. Split by time_block_id (contiguous temporal), NOT random shuffle.
3. Ratios: 70/15/15 from config.
4. Verify no time_block_id appears in more than one split.
5. Fit StandardScaler on training split only; save scaler to disk.
6. Apply scaler to val/test; save all scaled splits.
7. Report per-device, per-split window counts.
"""

import os
import csv
import pickle
import yaml
import numpy as np

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE_SUBSET  = config["dataset"]["device_subset"]
TRAIN_RATIO    = config["splits"]["train_ratio"]
VAL_RATIO      = config["splits"]["val_ratio"]
TEST_RATIO     = config["splits"]["test_ratio"]

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR  = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_feature_csv(filename):
    """Load a feature CSV, returning rows as list of dicts and feature column names."""
    filepath = os.path.join(FEATURES_DIR, filename)
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    meta_cols = ["device_label", "window_id", "time_block_id"]
    feature_cols = [c for c in fieldnames if c not in meta_cols]
    return rows, feature_cols


def assign_splits(all_block_ids):
    """
    Assign time blocks to train/val/test using a contiguous temporal split.
    Returns: dict mapping block_id -> split_name, and the three block lists.
    """
    sorted_blocks = sorted(all_block_ids)
    n = len(sorted_blocks)

    n_train = round(n * TRAIN_RATIO)
    n_val   = round(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    # Ensure every split has at least one block
    if n_test < 1:
        n_test = 1
        n_val  = min(n_val, n - n_train - 1)
    if n_val < 1:
        n_val   = 1
        n_train = n - n_val - n_test

    train_blocks = sorted_blocks[:n_train]
    val_blocks   = sorted_blocks[n_train:n_train + n_val]
    test_blocks  = sorted_blocks[n_train + n_val:]

    assignment = {}
    for b in train_blocks:
        assignment[b] = "train"
    for b in val_blocks:
        assignment[b] = "val"
    for b in test_blocks:
        assignment[b] = "test"

    return assignment, train_blocks, val_blocks, test_blocks


def verify_no_overlap(train_blocks, val_blocks, test_blocks):
    """Verify no time_block_id appears in more than one split."""
    train_set = set(train_blocks)
    val_set   = set(val_blocks)
    test_set  = set(test_blocks)

    issues = []
    if train_set & val_set:
        issues.append(f"Train ∩ Val: {train_set & val_set}")
    if train_set & test_set:
        issues.append(f"Train ∩ Test: {train_set & test_set}")
    if val_set & test_set:
        issues.append(f"Val ∩ Test: {val_set & test_set}")

    return len(issues) == 0, issues


class StandardScaler:
    """Simple StandardScaler — fit on training data only."""

    def __init__(self):
        self.mean_ = None
        self.std_  = None
        self.feature_names_ = None

    def fit(self, data, feature_names):
        self.mean_ = np.mean(data, axis=0)
        self.std_  = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1.0          # avoid division by zero
        self.feature_names_ = feature_names
        return self

    def transform(self, data):
        return (data - self.mean_) / self.std_

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_,
                         "feature_names": self.feature_names_}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        scaler = cls()
        scaler.mean_ = state["mean"]
        scaler.std_  = state["std"]
        scaler.feature_names_ = state["feature_names"]
        return scaler


def split_and_scale(rows, feature_cols, block_assignment, scaler_name):
    """
    Split rows by time block, fit scaler on train, apply to all splits.
    Returns split data dict and the fitted scaler.
    """
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        split = block_assignment.get(int(row["time_block_id"]))
        if split:
            splits[split].append(row)

    def to_matrix(row_list):
        return np.array([[float(r[c]) for c in feature_cols] for r in row_list])

    train_X = to_matrix(splits["train"])
    val_X   = to_matrix(splits["val"])
    test_X  = to_matrix(splits["test"])

    scaler = StandardScaler()
    scaler.fit(train_X, feature_cols)

    train_X_scaled = scaler.transform(train_X)
    val_X_scaled   = scaler.transform(val_X)
    test_X_scaled  = scaler.transform(test_X)

    scaler_path = os.path.join(RESULTS_DIR, f"{scaler_name}_scaler.pkl")
    scaler.save(scaler_path)

    return {
        "train": {"rows": splits["train"], "X": train_X, "X_scaled": train_X_scaled},
        "val":   {"rows": splits["val"],   "X": val_X,   "X_scaled": val_X_scaled},
        "test":  {"rows": splits["test"],  "X": test_X,  "X_scaled": test_X_scaled},
    }, scaler, scaler_path


def save_split_data(split_data, feature_cols, prefix):
    """Save scaled feature data per split."""
    meta_cols = ["device_label", "window_id", "time_block_id", "split"]
    all_cols  = meta_cols + feature_cols

    for split_name in ["train", "val", "test"]:
        rows     = split_data[split_name]["rows"]
        X_scaled = split_data[split_name]["X_scaled"]

        filepath = os.path.join(FEATURES_DIR, f"{prefix}_{split_name}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(all_cols)
            for i, row in enumerate(rows):
                meta  = [row["device_label"], row["window_id"],
                         row["time_block_id"], split_name]
                feats = [f"{v:.9f}" for v in X_scaled[i]]
                writer.writerow(meta + feats)


def save_split_assignment(block_assignment, train_blocks, val_blocks, test_blocks):
    """Save the split assignment mapping to CSV and pickle."""
    filepath = os.path.join(RESULTS_DIR, "split_assignment.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_block_id", "split"])
        for block_id in sorted(block_assignment.keys()):
            writer.writerow([block_id, block_assignment[block_id]])

    pkl_path = os.path.join(RESULTS_DIR, "split_assignment.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "assignment":   block_assignment,
            "train_blocks": train_blocks,
            "val_blocks":   val_blocks,
            "test_blocks":  test_blocks,
        }, f)

    return filepath, pkl_path


def main():
    print("=" * 80)
    print("  STEP 6: LEAKAGE-SAFE TRAIN/VAL/TEST SPLIT + SCALING")
    print(f"  Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Split method: contiguous temporal by time_block_id")
    print("=" * 80)

    # ── Load raw statistical features ──────────────────────────────────
    print("\n  Loading raw statistical features...")
    rows, feature_cols = load_feature_csv("raw_stat_features.csv")
    print(f"  {len(rows):,} rows, {len(feature_cols)} features")

    # ── Collect all unique time blocks ─────────────────────────────────
    all_blocks = sorted({int(r["time_block_id"]) for r in rows})
    print(f"\n  Total unique time blocks: {len(all_blocks)}")
    print(f"  Block range: {all_blocks[0]} to {all_blocks[-1]}")

    # ── Assign splits ──────────────────────────────────────────────────
    block_assignment, train_blocks, val_blocks, test_blocks = assign_splits(all_blocks)

    print(f"\n  Block assignment:")
    print(f"    Train: {len(train_blocks)} blocks "
          f"({len(train_blocks)/len(all_blocks)*100:.1f}%)")
    print(f"    Val:   {len(val_blocks)} blocks "
          f"({len(val_blocks)/len(all_blocks)*100:.1f}%)")
    print(f"    Test:  {len(test_blocks)} blocks "
          f"({len(test_blocks)/len(all_blocks)*100:.1f}%)")

    # ── No-overlap verification ────────────────────────────────────────
    passed, issues = verify_no_overlap(train_blocks, val_blocks, test_blocks)
    if passed:
        print("\n  [PASS] No time_block_id appears in more than one split.")
    else:
        print("\n  [FAIL] Overlap detected!")
        for issue in issues:
            print(f"    {issue}")
        return

    if max(train_blocks) < min(val_blocks) < min(test_blocks):
        print("  [PASS] Splits are temporally contiguous (train < val < test).")
    else:
        print("  [WARN] Splits are not temporally contiguous.")

    # ── Save split assignment ──────────────────────────────────────────
    csv_path, pkl_path = save_split_assignment(
        block_assignment, train_blocks, val_blocks, test_blocks
    )
    print(f"\n  Split assignment saved: {csv_path}")

    # ── Split and scale ────────────────────────────────────────────────
    print(f"\n  Scaling raw statistical features...")
    split_data, scaler, scaler_path = split_and_scale(
        rows, feature_cols, block_assignment, "raw_stat"
    )
    print(f"  Scaler fit on {len(split_data['train']['rows']):,} training rows.")
    print(f"  Scaler saved: {scaler_path}")
    save_split_data(split_data, feature_cols, "raw_scaled")

    # ── Per-device distribution ────────────────────────────────────────
    print(f"\n  {'='*80}")
    print("  PER-DEVICE WINDOW DISTRIBUTION ACROSS SPLITS")
    print(f"  {'='*80}")

    device_split_counts = {}
    for split_name in ["train", "val", "test"]:
        for row in split_data[split_name]["rows"]:
            key = (row["device_label"], split_name)
            device_split_counts[key] = device_split_counts.get(key, 0) + 1

    print(f"  {'Device':<28} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print(f"  {'-'*62}")
    for device in DEVICE_SUBSET:
        tr = device_split_counts.get((device, "train"), 0)
        va = device_split_counts.get((device, "val"),   0)
        te = device_split_counts.get((device, "test"),  0)
        print(f"  {device:<28} {tr:>10,} {va:>10,} {te:>10,} {tr+va+te:>10,}")

    tot_tr = sum(device_split_counts.get((d, "train"), 0) for d in DEVICE_SUBSET)
    tot_va = sum(device_split_counts.get((d, "val"),   0) for d in DEVICE_SUBSET)
    tot_te = sum(device_split_counts.get((d, "test"),  0) for d in DEVICE_SUBSET)
    print(f"  {'-'*62}")
    print(f"  {'TOTAL':<28} {tot_tr:>10,} {tot_va:>10,} {tot_te:>10,} "
          f"{tot_tr+tot_va+tot_te:>10,}")

    # ── Scaling sanity check ───────────────────────────────────────────
    train_scaled = split_data["train"]["X_scaled"]
    print(f"\n  Scaling verification (train should be ~mean=0, ~std=1):")
    print(f"    Mean range: [{np.mean(train_scaled, axis=0).min():.2e}, "
          f"{np.mean(train_scaled, axis=0).max():.2e}]")
    print(f"    Std  range: [{np.std(train_scaled, axis=0).min():.4f}, "
          f"{np.std(train_scaled, axis=0).max():.4f}]")

    print(f"\n  Step 6 complete.")


if __name__ == "__main__":
    main()
