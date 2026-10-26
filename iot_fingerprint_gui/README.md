# Encrypted IoT Device Fingerprinting Web GUI

A responsive, production-ready full-stack web application wired directly to the trained 100-tree Random Forest model for encrypted network traffic timing analysis.

---

## 🔒 Isolation Rule Compliance
This entire application and its assets reside strictly inside `iot_fingerprint_gui/`.
* **Zero side-effects on original project:** Existing code, trained model weights, datasets, and scripts in the repository root remain completely unmodified and read-only.
* **Self-contained dependencies:** Standalone `requirements.txt` for Python and dedicated `package.json` for React/Vite.
* **Isolated assets:** Model weights, feature scalers, and replay samples are copied into `iot_fingerprint_gui/assets/` with full provenance documented in `assets/metadata.json`.

---

## 🚀 Quickstart: Running the Application

### 1. Start the FastAPI Backend (Port 8765)
In a terminal, navigate to the project directory and run:
```bash
# Using Python
python -m uvicorn iot_fingerprint_gui.backend.main:app --host 127.0.0.1 --port 8765
```
* Backend API will be available at: `http://127.0.0.1:8765`
* Interactive API Documentation (Swagger): `http://127.0.0.1:8765/docs`

### 2. Start the React Frontend (Port 5173)
In a second terminal:
```bash
cd iot_fingerprint_gui/frontend
npm run dev
```
* Open your browser at: `http://localhost:5173`

---

## 📊 Dataset & Model Ground Truth

* **Model Type:** Random Forest Classifier (100 decision trees)
* **Dataset:** UNSW Sydney Smart Home Benchmark Dataset (Nov 2016)
* **Pipeline:** Packet Arrival Timestamps $\to$ Inter-Arrival Times (IAT) $\to$ $\log(1 + \text{IAT})$ $\to$ Non-overlapping 128-Packet Windows $\to$ 13 Statistical Moments $\to$ Z-Score Standardization $\to$ `predict_proba`.
* **The 13 Statistical Features:**
  `raw_mean`, `raw_std`, `raw_min`, `raw_max`, `raw_median`, `raw_skew`, `raw_kurtosis`, `raw_p25`, `raw_p75`, `raw_p95`, `raw_p99`, `raw_energy`, `raw_entropy`

### The 5 Trained (Known) Devices
The model was trained exclusively on these 5 devices:
1. **Amazon Echo** (`44:65:0d:56:cc:d3`) &mdash; Smart Voice Speaker
2. **Belkin WeMo Motion Sensor** (`ec:1a:59:83:28:11`) &mdash; PIR Motion Detector
3. **Philips Hue Bridge** (`00:17:88:24:76:ff`) &mdash; ZigBee Gateway
4. **Samsung SmartCam** (`00:16:6c:ab:6b:88`) &mdash; IP Security Camera
5. **TP-Link Smart Plug** (`50:c7:bf:00:56:39`) &mdash; Wi-Fi Power Switch

### The 22 Unknown Devices (Never Seen During Training)
All 22 remaining devices in the dataset are treated as out-of-distribution (unknown) inputs:
* **15 Active Unknown Devices (Real traffic extracted):**
  Awair Air Quality, Belkin WeMo Switch, Canary Camera, HP Envy Printer, LiFX Smart Bulb, Nest DropCam, Nest Protect Smoke Alarm, Netatmo Weather Station, Netatmo Welcome Camera, PixStar Photo Frame, Samsung SmartThings Hub, Triby Speaker, Withings Sleep Sensor, Withings Smart Scale, iHome Smart Plug.
* **7 Excluded / Offline Devices (0 packets in Nov 2016 trace):**
  August Doorbell Cam, Belkin NetCam, Blipcare BP Meter, Hello Barbie, Ring Doorbell, TP-Link Cloud Camera, Withings Baby Monitor.

---

## 🎯 Calibrated Decision Rule & Empirical Performance

Every 128-packet window is evaluated by the shared decision function:
$$\text{top\_prob} = \max_{c \in \text{Classes}} P(y = c \mid x)$$
$$\text{Verdict} = \begin{cases} \text{KNOWN (Label } = c^*), & \text{if } \text{top\_prob} \ge \tau \\ \text{UNKNOWN (Closest } = c^*), & \text{if } \text{top\_prob} < \tau \end{cases}$$

### Empirical Calibration (No Data Leakage)
* **Threshold ($\tau$):** **`0.8400`**
* **Calibration Set:** Derived exclusively from the 5th percentile of max class probability on the **1,071 validation windows** of the 5 known devices.
* **Measured Test False Alarm Rate:** **`3.94%`** (42 / 1,067 test windows incorrectly flagged below 0.8400 on known held-out traffic).
* **Measured Unknown Detection Rate:** **`67.58%`** (3,370 / 4,987 total unknown windows correctly flagged as low confidence).

### High-Rejection Devices vs Confident Closed-Set Failures
* **High Rejection ($\ge 92\%$ flagged UNKNOWN):**
  Nest DropCam (100%), Netatmo Weather (100%), PixStar Photo Frame (100%), iHome (100%), Withings Smart Scale (100%), Nest Protect (100%), Triby Speaker (99.6%), LiFX Bulb (98.8%), HP Printer (94.8%), Netatmo Welcome (92.8%), Canary Camera (92.2%).
* **Honest Limitations & False Matches:**
  * **Belkin WeMo Switch (30.8% flagged):** Mislabeled as Belkin WeMo Motion Sensor (96% of windows, avg confidence 0.8612) due to identical firmware UPnP SSDP heartbeat intervals.
  * **Awair Air Quality (1.6% flagged):** Periodic telemetry bursts align with the Amazon Echo keepalive cycle.
  * **Withings Sleep Sensor (0.6% flagged):** Mislabeled as Amazon Echo.

---

## 🌐 Features of the GUI

### Tab 1: Known Devices (5)
* Dedicated card per trained hardware with an interactive **TOGGLE** switch.
* Replays held-out test windows live via WebSockets.
* Displays live 5-class probability bars, 128-packet IAT timeline chart, 13 statistical feature moments, and running accuracy.
* Global panel with confusion matrix and Gini feature importances.

### Tab 2: Unknown Devices (22)
* Filterable directory of all 22 unseen hardware devices.
* Toggle switch to stream real traffic through the model and observe closed-set model responses.
* Dynamic threshold slider with live updates and detection/false-alarm trade-off estimation.
* Complete evaluation benchmark table showing tested windows, dominant mislabeled profile, and failure modes.

### Tab 3: Test Your Own Device
* Three flexible input modes:
  1. **Upload File:** `.pcap`, `.pcapng`, or `.csv`
  2. **Live Promiscuous Sniffing:** Interface selection with optional IP/MAC BPF filters
  3. **Paste Timestamps:** Space- or comma-separated list of packet timestamps
* Per-window probability inspection and device-level aggregate verdict via majority vote.
* Export full evaluation results as **CSV** or **JSON**.
* **Safety Notice:** Payloads are never decrypted or read; only arrival timestamps are processed.

---

## ⚙️ Configuration (.env)
Configured via `iot_fingerprint_gui/.env`:
```ini
BACKEND_PORT=8765
BACKEND_HOST=127.0.0.1
FRONTEND_PORT=5173
VITE_API_URL=http://localhost:8765
VITE_WS_URL=ws://localhost:8765
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```
