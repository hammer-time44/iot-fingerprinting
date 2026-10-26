# IoT Device Fingerprinting from Encrypted Traffic

Passive identification of IoT devices from network traffic timing,
without inspecting packet payloads or storing any local identifiers
(MAC addresses, IP addresses).

Given a sequence of packets captured from a home network, the system
extracts inter-arrival time (IAT) statistics from fixed-length windows
and uses a Random Forest to classify which of the five known devices
produced them.

---

## Devices

| Device | MAC (source file only) |
|---|---|
| AmazonEcho | 44:65:0d:56:cc:d3 |
| BelkinWemoMotionSensor | ec:1a:59:83:28:11 |
| PhilipsHue | 00:17:88:2b:9a:25 |
| SamsungCamera | 00:16:6c:ab:6b:88 |
| TPLinkSmartPlug | 50:c7:bf:00:56:39 |

**Capture window:** November 22–24, 2016 (3 days).

---

## Pipeline

Each step is an independently runnable script. Run them in order.

```
Step 2  src/preprocessing.py   PCAP → per-device CSV (timestamp, packet_size, direction)
Step 3  src/iat.py             Inter-arrival time extraction
Step 4  src/windowing.py       log1p transform + 128-packet non-overlapping windows
Step 5  src/features.py        13 raw statistical features per window
Step 6  src/splits.py          Leakage-safe temporal train/val/test split + StandardScaler
Step 7  src/classifier.py      Random Forest training and evaluation
Step 8  src/openset.py         Open-set unknown device evaluation & thresholding
```

### Step 2 — Preprocessing

Reads raw PCAP files from `data/raw/`. For each packet belonging to a
target device, records only:
- `timestamp` (Unix seconds)
- `packet_size` (bytes)
- `direction` (0 = incoming, 1 = outgoing)

No MAC address, IP address, or payload is stored in the output.
Output: `data/processed/{device}.csv`

### Step 3 — IAT Extraction

Computes `IAT[i] = timestamp[i+1] − timestamp[i]` for consecutive
packets within each device's stream.
Output: `data/processed/{device}_iat.csv`

### Step 4 — Windowing

Applies `log(1 + IAT)` to compress the multi-order-of-magnitude
range of timing values, then slices each device's sequence into
non-overlapping 128-packet windows. Each window is tagged with a
1-hour `time_block_id` used in Step 6 to prevent temporal leakage.
Incomplete trailing windows are dropped.
Output: `data/features/{device}_windows.csv`

### Step 5 — Feature Extraction

Computes 13 raw statistical features per window from the
log1p-transformed IAT values:

| Feature | Description |
|---|---|
| raw_mean | Mean IAT |
| raw_std | Standard deviation |
| raw_min / raw_max | Range |
| raw_median | Median |
| raw_skew | Fisher skewness |
| raw_kurtosis | Excess kurtosis |
| raw_p25 / raw_p75 / raw_p95 / raw_p99 | Percentiles |
| raw_energy | Sum of squared values |
| raw_entropy | Shannon entropy on squared magnitudes |

Output: `data/features/raw_stat_features.csv`

### Step 6 — Splitting and Scaling

Splits windows by time block using a **contiguous temporal** split
(70% train / 15% val / 15% test). No time block appears in more than
one split. A `StandardScaler` is fit on the training split only and
applied to validation and test.

Output: `data/features/raw_scaled_{train,val,test}.csv`
        `results/raw_stat_scaler.pkl`
        `results/split_assignment.{csv,pkl}`

### Step 7 — Classification

Trains a Random Forest (100 trees, unlimited depth, 1 sample per leaf)
on the 13 scaled features. Evaluates on the validation split during
development, then touches the test split exactly once.

Reports:
- Accuracy and macro F1 on both val and test
- Per-device precision, recall, and F1
- Confusion matrices (PNG) saved to a timestamped `results/step7_*/`
  folder alongside the model and a config snapshot

### Step 8 — Open-Set / Unknown Device Detection

Extends the system to flag traffic as either one of the known devices or
as an **"Unknown Device"** when traffic does not sufficiently match the
known timing signatures.

**Decision Rule:**
For an incoming traffic window $x$, compute the prediction anomaly score:
$$S(x) = 1.0 - \max_{c \in \text{known}} P(y = c \mid x)$$
- If $S(x) > \tau$: flag as **`Unknown Device`** (insufficient match).
- Else: classify as $\hat{y} = \arg\max_c P(y = c \mid x)$.

**Threshold Calibration:**
The decision threshold $\tau$ is selected strictly from the **known validation split**
at a configured percentile (e.g., 95th percentile targeting a ~5% false alarm budget,
or 99th percentile targeting a ~1% budget). It is **never** tuned on the test set.

**Leave-One-Device-Out Empirical Benchmark:**
To rigorously test unknown detection without assuming it works, each device is held out
from training in turn to simulate a genuine unknown transmitter. Evaluated on the unseen test split:

| Held-Out Unknown Device | Test Windows | Val 95%ile Thresh (5% target FPR) | Val 99%ile Thresh (1% target FPR) |
|---|:---:|:---:|:---:|
| | | Known FPR &nbsp; Unknown TPR &nbsp; Acc(Ver) | Known FPR &nbsp; Unknown TPR &nbsp; Acc(Ver) |
| **AmazonEcho** | 204 | 3.8% &emsp;&emsp; **93.1%** &emsp;&emsp; 100.0% | 0.3% &emsp;&emsp; **53.4%** &emsp;&emsp; 99.8% |
| **BelkinWemoMotionSensor** | 423 | 4.2% &emsp;&emsp; **66.2%** &emsp;&emsp; 100.0% | 1.4% &emsp;&emsp; **14.2%** &emsp;&emsp; 100.0% |
| **PhilipsHue** | 242 | 4.8% &emsp;&emsp; **88.4%** &emsp;&emsp; 100.0% | 1.6% &emsp;&emsp; **66.1%** &emsp;&emsp; 99.9% |
| **SamsungCamera** | 189 | 4.0% &emsp;&emsp; **100.0%** &emsp;&emsp; 100.0% | 1.7% &emsp;&emsp; **70.9%** &emsp;&emsp; 100.0% |
| **TPLinkSmartPlug** | 9 | 4.1% &emsp;&emsp; **100.0%** &emsp;&emsp; 100.0% | 1.5% &emsp;&emsp; **100.0%** &emsp;&emsp; 99.8% |
| **Average Across Devices** | — | **4.2%** &emsp;&emsp; **89.6%** &emsp;&emsp; — | **1.3%** &emsp;&emsp; **60.9%** &emsp;&emsp; — |

*Acc(Ver) is the classification accuracy on test windows accepted as known.*

**Open-Set Findings & Limitations:**
- **Outlier timing profiles are reliably detected:** Devices with distinct timing behavior (such as `TPLinkSmartPlug`'s ~4-minute idle heartbeats and `SamsungCamera`'s continuous telemetry bursts) achieve **100% unknown detection**.
- **Overlapping profiles are harder to isolate:** Devices with timing characteristics that closely resemble other high-traffic smart devices (notably `BelkinWemoMotionSensor`) achieve **66.2%** detection at a 5% false alarm budget, dropping to **14.2%** at a 1% budget.
- **No network blocking:** This stage performs passive detection and flagging only; network access control or blocking is not enforced.

---

## Results Summary

| Metric | Closed-Set (5 Known Devices) | Open-Set (Leave-One-Out Average) |
|---|:---:|:---:|
| Test Accuracy / Known Acc | 99.3% | 100.0% (on verified known windows) |
| Test Macro F1 | 99.4% | — |
| Unknown Detection Rate (TPR @ 95%ile) | N/A (Closed-Set) | **89.6%** (66.2% – 100.0%) |
| Known False Alarm Rate (FPR @ 95%ile) | 0.0% | **4.2%** (controlled to ~5% target) |


---

## Setup

```bash
pip install -r requirements.txt
```

Raw PCAP files are not included in the repository (too large). Place
them in `data/raw/` before running Step 2. Steps 3–7 can be run on
the pre-processed data already present in `data/processed/` and
`data/features/`.

### Run order

```bash
python src/preprocessing.py   # requires data/raw/*.pcap
python src/iat.py
python src/windowing.py
python src/features.py
python src/splits.py
python src/classifier.py
python src/openset.py
```

Each script is self-contained and reads `configs/config.yaml`.

---

## Interactive Presentation Website

To launch the presentation website with interactive benchmark results, timing charts, and live inference on saved test samples:

```bash
streamlit run app.py
```

---

## Repository Layout

```
app.py                      Interactive presentation website (Streamlit)
configs/
  config.yaml               Pipeline configuration
data/
  raw/                      Raw PCAP files (not in git)
  processed/                Cleaned per-device CSVs and IAT sequences
  features/                 Windowed data, features, and scaled splits
results/
  raw_stat_scaler.pkl       Fitted StandardScaler
  split_assignment.csv      Time-block → split mapping
  step7_*/                  Timestamped run: model, confusion matrices, summary
src/
  preprocessing.py          Step 2
  iat.py                    Step 3
  windowing.py              Step 4
  features.py               Step 5 (raw statistical features)
  splits.py                 Step 6
  classifier.py             Step 7
  openset.py                Step 8 (open-set unknown detection evaluation)
requirements.txt
```
