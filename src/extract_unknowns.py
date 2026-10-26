"""
Extract, window, and compute features for unknown devices in the UNSW dataset
matching the exact capture window (Nov 22-24, 2016) and the exact 13-feature pipeline.
"""
import os
import sys
import datetime
import time
import dpkt
import numpy as np
import pandas as pd

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Candidate PCAP directory
PCAP_DIR = r"C:\Users\Murali\Downloads\pcaps\pcaps"
if not os.path.exists(PCAP_DIR):
    PCAP_DIR = r"C:\Users\Murali\Desktop\CN IDE\iot_fingerprinting\data\raw\pacps\pcaps"

DATE_START = datetime.datetime(2016, 11, 22, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
DATE_END = datetime.datetime(2016, 11, 25, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()

KNOWN_DEVICES = {
    "AmazonEcho",
    "BelkinWemoMotionSensor",
    "PhilipsHue",
    "SamsungCamera",
    "TPLinkSmartPlug",
}

# The 13 statistical feature function (exact copy from src/features.py)
def shannon_entropy(values):
    sq = np.array(values) ** 2
    total = np.sum(sq)
    if total == 0:
        return 0.0
    probs = sq / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def compute_raw_stats(window_values):
    arr = np.array(window_values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    median_val = float(np.median(arr))

    if std_val > 0:
        skew_val = float(np.mean(((arr - mean_val) / std_val) ** 3))
        kurt_val = float(np.mean(((arr - mean_val) / std_val) ** 4) - 3.0)
    else:
        skew_val = 0.0
        kurt_val = 0.0

    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    energy = float(np.sum(arr ** 2))
    entropy = shannon_entropy(arr)

    return {
        "raw_mean": mean_val,
        "raw_std": std_val,
        "raw_min": min_val,
        "raw_max": max_val,
        "raw_median": median_val,
        "raw_skew": skew_val,
        "raw_kurtosis": kurt_val,
        "raw_p25": p25,
        "raw_p75": p75,
        "raw_p95": p95,
        "raw_p99": p99,
        "raw_energy": energy,
        "raw_entropy": entropy,
    }

def mac_str(b):
    return ":".join(f"{x:02x}" for x in b)

def process_pcap_device(fname):
    fpath = os.path.join(PCAP_DIR, fname)
    parts = fname.replace(".pcapng", "").replace(".pcap", "").split("_")
    dev_name = parts[0]
    dev_mac = ":".join(parts[1][i:i+2] for i in range(0, len(parts[1]), 2)).lower() if len(parts) > 1 else ""

    if dev_name in KNOWN_DEVICES:
        return None

    t0 = time.time()
    timestamps = []

    # Known offline devices in Nov 2016 (from UNSW dataset summary dates)
    OFFLINE_NOV2016 = {
        "AugustDoorBell", "BelkinCamera", "BlipCareBPMeter", "HelloBarbie",
        "RingDoorBell", "TPLinkCamera", "WithingsBabyMonitor"
    }
    if dev_name in OFFLINE_NOV2016:
        print(f"Skipping {dev_name:<24} (MAC: {dev_mac}): offline / 0 packets during Nov 22-24, 2016 capture window.", flush=True)
        return {
            "device": dev_name,
            "mac": dev_mac,
            "packets": 0,
            "windows": 0,
            "status": "offline_in_capture_window",
            "reason": "Device was offline or not captured during Nov 22-24, 2016",
        }

    with open(fpath, "rb") as f:
        try:
            reader = dpkt.pcapng.Reader(f)
        except Exception:
            f.seek(0)
            reader = dpkt.pcap.Reader(f)

        consecutive_after_window = 0
        for ts, buf in reader:
            if ts < DATE_START:
                continue
            if ts >= DATE_END:
                consecutive_after_window += 1
                if consecutive_after_window > 1000:
                    break
                continue
            else:
                consecutive_after_window = 0

            if len(buf) < 14:
                continue
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                src_m = mac_str(eth.src)
                dst_m = mac_str(eth.dst)
                if src_m == dev_mac or dst_m == dev_mac:
                    timestamps.append(ts)
            except Exception:
                continue

    pkt_count = len(timestamps)
    elapsed = time.time() - t0

    if pkt_count < 129:
        print(f"Skipping {dev_name:<24} (MAC: {dev_mac}): {pkt_count} packets in window (too few for 128-pkt window) [{elapsed:.1f}s]", flush=True)
        return {
            "device": dev_name,
            "mac": dev_mac,
            "packets": pkt_count,
            "windows": 0,
            "status": "skipped_too_few_packets",
            "reason": f"Only {pkt_count} packets captured (requires at least 129 packets)",
        }

    # Sort timestamps
    timestamps = np.sort(timestamps)
    iats = np.diff(timestamps)
    log1p_iats = np.log1p(np.maximum(0.0, iats))

    # Slice into 128-packet windows (each window has 128 IAT values)
    n_windows = len(log1p_iats) // 128
    if n_windows < 1:
        return {
            "device": dev_name,
            "mac": dev_mac,
            "packets": pkt_count,
            "windows": 0,
            "status": "skipped_too_few_windows",
            "reason": "Fewer than 1 complete 128-packet window",
        }

    # Cap to max 500 windows per device if massive to keep evaluation fast & balanced
    max_eval_windows = min(n_windows, 500)

    feature_rows = []
    window_rows = []

    for w_idx in range(max_eval_windows):
        w_values = log1p_iats[w_idx * 128 : (w_idx + 1) * 128]
        stats = compute_raw_stats(w_values)
        stats["device_label"] = dev_name
        stats["device_mac"] = dev_mac
        stats["window_id"] = w_idx
        feature_rows.append(stats)

        # Store first 5 windows for raw waveform visualizer
        if w_idx < 5:
            w_dict = {"device": dev_name, "mac": dev_mac, "window_id": w_idx}
            for i, val in enumerate(w_values):
                w_dict[f"iat_{i}"] = float(val)
            window_rows.append(w_dict)

    print(f"PROCESSED: {dev_name:<22} MAC: {dev_mac} | Packets: {pkt_count:>7,} | Windows: {n_windows:>4} (evaluated: {max_eval_windows}) [{elapsed:.1f}s]", flush=True)
    return {
        "device": dev_name,
        "mac": dev_mac,
        "packets": pkt_count,
        "windows": n_windows,
        "evaluated_windows": max_eval_windows,
        "status": "processed",
        "feature_rows": feature_rows,
        "window_rows": window_rows,
    }

def main():
    print("=" * 80, flush=True)
    print("EXTRACTING UNKNOWN DEVICES FROM RAW PCAPS", flush=True)
    print(f"Capture window: 2016-11-22 00:00:00 to 2016-11-25 00:00:00 UTC", flush=True)
    print(f"PCAP source: {PCAP_DIR}", flush=True)
    print("=" * 80, flush=True)

    files = [f for f in os.listdir(PCAP_DIR) if f.endswith(".pcap") or f.endswith(".pcapng")]
    files.sort(key=lambda x: os.path.getsize(os.path.join(PCAP_DIR, x)))

    all_features = []
    all_window_rows = []
    summary_report = []

    for fname in files:
        res = process_pcap_device(fname)
        if res is None:
            continue
        summary_report.append({
            "Device": res["device"],
            "MAC": res["mac"],
            "Packets (Nov 22-24)": res["packets"],
            "128-Pkt Windows": res["windows"],
            "Evaluated Windows": res.get("evaluated_windows", 0),
            "Status": res["status"],
            "Notes": res.get("reason", "Ready for inference"),
        })
        if res.get("feature_rows"):
            all_features.extend(res["feature_rows"])
        if res.get("window_rows"):
            all_window_rows.extend(res["window_rows"])

    if all_features:
        df_feat = pd.DataFrame(all_features)
        feat_path = os.path.join(FEATURES_DIR, "unknown_devices_features.csv")
        df_feat.to_csv(feat_path, index=False)
        print(f"\nSaved {len(df_feat)} unknown device feature windows to {feat_path}", flush=True)

    if all_window_rows:
        df_win = pd.DataFrame(all_window_rows)
        win_path = os.path.join(FEATURES_DIR, "unknown_devices_windows.csv")
        df_win.to_csv(win_path, index=False)
        print(f"Saved {len(df_win)} unknown device raw window traces to {win_path}", flush=True)

    df_summary = pd.DataFrame(summary_report)
    summary_path = os.path.join(RESULTS_DIR, "unknown_devices_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved device summary to {summary_path}", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("SUMMARY OF UNKNOWN DEVICES IN UNSW BENCHMARK DATASET:", flush=True)
    print("=" * 95, flush=True)
    print(df_summary.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
