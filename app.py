"""
Encrypted IoT Device Fingerprinting — Interactive Presentation Dashboard
========================================================================
A student-friendly, polished presentation website that demonstrates passive
device fingerprinting on encrypted network traffic using Inter-Arrival Times (IAT),
statistical summaries, Random Forest classification, and validation-calibrated
open-set unknown device detection.

To run:
    streamlit run app.py
"""

import os
import pickle
import json
import random
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & METADATA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT Device Fingerprinting | Encrypted Traffic Analysis",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJ_ROOT, "results", "step7_20260917_144006")
OPENSET_JSON_PATH = os.path.join(PROJ_ROOT, "results", "openset_results.json")
FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
MODEL_PATH = os.path.join(RESULTS_DIR, "rf_raw_model.pkl")
RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, "results_summary.json")

DEVICE_NAMES = [
    "AmazonEcho",
    "BelkinWemoMotionSensor",
    "PhilipsHue",
    "SamsungCamera",
    "TPLinkSmartPlug",
]

DEVICE_FRIENDLY = {
    "AmazonEcho": "Amazon Echo (Smart Speaker)",
    "BelkinWemoMotionSensor": "Belkin WeMo (Motion Sensor)",
    "PhilipsHue": "Philips Hue (Lighting Bridge)",
    "SamsungCamera": "Samsung SmartCam (Security Camera)",
    "TPLinkSmartPlug": "TP-Link Smart Plug (Power Switch)",
}

DEVICE_ICONS = {
    "AmazonEcho": "🔊",
    "BelkinWemoMotionSensor": "🏃",
    "PhilipsHue": "💡",
    "SamsungCamera": "📷",
    "TPLinkSmartPlug": "🔌",
}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        color: #f8fafc;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(129, 140, 248, 0.4);
        color: #c7d2fe;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.6rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        max-width: 880px;
        line-height: 1.6;
    }

    .scope-banner {
        background: rgba(99, 102, 241, 0.12);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 0.95rem 1.25rem;
        margin-bottom: 1.8rem;
        color: #e0e7ff;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .scope-banner strong {
        color: #a5b4fc;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-card-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.1;
    }
    .metric-card-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    .flow-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.1rem;
        height: 100%;
        position: relative;
    }
    .flow-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #6366f1;
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
    }
    .flow-title {
        font-weight: 700;
        font-size: 1.02rem;
        color: #f1f5f9;
        margin-bottom: 0.4rem;
    }
    .flow-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    .flow-tag {
        display: inline-block;
        margin-top: 0.6rem;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }

    .pred-box {
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .pred-match {
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.4);
    }
    .pred-mismatch {
        background: rgba(239, 68, 68, 0.12);
        border-color: rgba(239, 68, 68, 0.4);
    }
    .pred-unknown {
        background: rgba(245, 158, 11, 0.14);
        border-color: rgba(245, 158, 11, 0.45);
    }

    .chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .chip-green  { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .chip-blue   { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .chip-amber  { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .chip-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_rf_model():
    """Load the pre-trained Random Forest model trained on raw statistical features."""
    candidate_paths = [
        os.path.join(PROJ_ROOT, "results", "step7_20261026_151020", "rf_model.pkl"),
        os.path.join(PROJ_ROOT, "results", "step7_20260917_144006", "rf_raw_model.pkl"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return None

@st.cache_data
def load_saved_results_json():
    """Load the saved benchmark results summary JSON."""
    if not os.path.exists(RESULTS_JSON_PATH):
        return None
    with open(RESULTS_JSON_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_openset_json():
    """Load the empirical leave-one-device-out open-set results."""
    if not os.path.exists(OPENSET_JSON_PATH):
        return None
    with open(OPENSET_JSON_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_test_and_val_data():
    """Load the scaled test and val feature CSVs."""
    test_path = os.path.join(FEATURES_DIR, "raw_scaled_test.csv")
    val_path = os.path.join(FEATURES_DIR, "raw_scaled_val.csv")
    unscaled_path = os.path.join(FEATURES_DIR, "raw_stat_features.csv")

    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None
    df_val = pd.read_csv(val_path) if os.path.exists(val_path) else None
    df_unscaled = pd.read_csv(unscaled_path) if os.path.exists(unscaled_path) else None

    return df_test, df_val, df_unscaled

@st.cache_data
def load_scaler():
    """Load fitted feature scaler state."""
    scaler_path = os.path.join(PROJ_ROOT, "results", "raw_stat_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_unknown_devices_data():
    """Load pre-extracted features and summary for all unknown devices in the dataset."""
    feat_path = os.path.join(FEATURES_DIR, "unknown_devices_features.csv")
    summary_path = os.path.join(PROJ_ROOT, "results", "unknown_devices_summary.csv")
    eval_table_path = os.path.join(PROJ_ROOT, "results", "unknown_evaluation_table.csv")

    df_feats = pd.read_csv(feat_path) if os.path.exists(feat_path) else None
    df_summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    df_eval = pd.read_csv(eval_table_path) if os.path.exists(eval_table_path) else None

    return df_feats, df_summary, df_eval

@st.cache_data
def load_window_sequence(device_name, window_id):
    """Load the 128 raw IAT values for a specific device window if available."""
    wpath = os.path.join(FEATURES_DIR, f"{device_name}_windows.csv")
    if not os.path.exists(wpath):
        return None
    try:
        df = pd.read_csv(wpath)
        row = df[df["window_id"] == window_id]
        if row.empty:
            return None
        iat_cols = [c for c in df.columns if c.startswith("iat_")]
        return row[iat_cols].values[0]
    except Exception:
        return None


# Load artifacts
rf_model = load_rf_model()
results_data = load_saved_results_json()
openset_data = load_openset_json()
df_test, df_val, df_unscaled = load_test_and_val_data()
scaler_dict = load_scaler()
df_unk_features, df_unk_summary, df_unk_eval = load_unknown_devices_data()

FEATURE_COLS = [
    "raw_mean", "raw_std", "raw_min", "raw_max", "raw_median",
    "raw_skew", "raw_kurtosis", "raw_p25", "raw_p75", "raw_p95",
    "raw_p99", "raw_energy", "raw_entropy"
]

# Extract benchmark numbers
raw_stats_summary = results_data.get("raw", {}) if results_data else {}
ref_test_acc = raw_stats_summary.get("test_accuracy", 0.9934)
ref_test_f1 = raw_stats_summary.get("test_f1_macro", 0.9938)
ref_val_acc = raw_stats_summary.get("val_accuracy", 0.9832)
ref_val_f1 = raw_stats_summary.get("val_f1_macro", 0.9866)
ref_per_class_test = raw_stats_summary.get("per_class", {})


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🔐 Network Security & Machine Learning</div>
    <div class="hero-title">Encrypted IoT Device Fingerprinting</div>
    <div class="hero-subtitle">
        Identifying known smart home hardware and flagging unknown devices purely from packet arrival rhythms 
        (Inter-Arrival Times) and statistical timing moments — without decrypting payloads or inspecting MAC/IP addresses.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD PRESENTATION DECK
# ─────────────────────────────────────────────────────────────────────────────
pptx_path = os.path.join(PROJ_ROOT, "project_overview.pptx")
if os.path.exists(pptx_path):
    d_col1, d_col2 = st.columns([3, 1])
    with d_col1:
        st.caption("📽️ Looking for the presentation deck? Download the 14-slide widescreen presentation deck (.pptx) with embedded real metrics and speaker notes.")
    with d_col2:
        with open(pptx_path, "rb") as pf:
            st.download_button(
                label="📥 Download Slides (.pptx)",
                data=pf.read(),
                file_name="project_overview.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_results, tab_example, tab_unknown, tab_guide = st.tabs([
    "🔍 1. How It Works (Workflow)",
    "📊 2. Benchmark Results & Exploration",
    "🎯 3. Try an Example (Saved Test Data)",
    "Unknown Device",
    "🎓 5. Presentation Cheat Sheet & FAQ",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: VISUAL WORKFLOW EXPLANATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Traffic Analysis Workflow")
    st.markdown(
        "Modern IoT traffic is encrypted with TLS/SSL. Deep packet inspection (DPI) fails because the payloads "
        "are unreadable, and MAC or IP addresses can be easily spoofed or masked by NAT routers. "
        "Our pipeline extracts **hardware and firmware timing signatures** through a 5-stage transformation:"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-number">1</div>
            <div class="flow-title">Packet Traffic</div>
            <div class="flow-desc">
                Captures raw packet events from network monitor. Strips all MAC & IP identifiers for strict privacy.
            </div>
            <div class="flow-tag">Step 2: preprocessing.py</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-number">2</div>
            <div class="flow-title">Time Deltas (IAT)</div>
            <div class="flow-desc">
                Calculates Inter-Arrival Time: &Delta;t = t<sub>i+1</sub> &minus; t<sub>i</sub>. Applies log(1 + IAT) to compress wide idle ranges.
            </div>
            <div class="flow-tag">Step 3: iat.py</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-number">3</div>
            <div class="flow-title">Fixed Windows</div>
            <div class="flow-desc">
                Slices packet sequences into 128-packet non-overlapping windows tagged with 1-hour time blocks.
            </div>
            <div class="flow-tag">Step 4: windowing.py</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-number">4</div>
            <div class="flow-title">Timing Summaries</div>
            <div class="flow-desc">
                Computes 13 statistical indicators per window: mean, std, percentiles (P25-P99), skewness, energy, entropy.
            </div>
            <div class="flow-tag">Step 5: features.py</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-number">5</div>
            <div class="flow-title">RF + Rejection</div>
            <div class="flow-desc">
                Random Forest outputs class probabilities. If confidence exceeds validation threshold &rarr; Known Device; else &rarr; Unknown.
            </div>
            <div class="flow-tag">Steps 7-8: classifier & openset</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Devices & Dataset Info
    st.subheader("The Five Target Devices")
    st.markdown("All data was captured from the authoritative **UNSW-IoTraffic** capture over a 3-day continuous window (Nov 22–24, 2016):")

    d_cols = st.columns(5)
    devices_info = [
        ("AmazonEcho", "Amazon Echo", "Smart Speaker", "High volume, frequent voice and cloud heartbeat transactions."),
        ("BelkinWemoMotionSensor", "Belkin WeMo", "Motion Sensor", "Very high traffic density, periodic passive polling spikes."),
        ("PhilipsHue", "Philips Hue", "Lighting Bridge", "Consistent periodic hub communication with light bulbs."),
        ("SamsungCamera", "Samsung SmartCam", "Security Camera", "High packet volume, steady streaming telemetry buffers."),
        ("TPLinkSmartPlug", "TP-Link Smart Plug", "Power Switch", "Low traffic volume; distinct ~4-minute extreme idle heartbeats."),
    ]

    for idx, (key, name, role, note) in enumerate(devices_info):
        with d_cols[idx]:
            icon = DEVICE_ICONS.get(key, "📱")
            st.markdown(f"""
            <div style="background: rgba(30,41,59,0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1rem; height: 100%;">
                <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">{icon}</div>
                <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{name}</div>
                <div style="font-size: 0.78rem; color: #818cf8; font-weight: 600; margin-bottom: 0.4rem;">{role}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">{note}</div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: SAVED BENCHMARK RESULTS & EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("Saved Reference Results")
    st.caption("Results saved from Step 7 Random Forest benchmark and Step 8 Open-Set Evaluation. Fitted on 70% train split, evaluated strictly on held-out temporal splits.")

    # High-level KPI Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{ref_test_acc * 100:.2f}%</div>
            <div class="metric-card-lbl">Test Accuracy</div>
            <div class="metric-card-sub">1,067 test windows (Touch once)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{ref_test_f1 * 100:.2f}%</div>
            <div class="metric-card-lbl">Test Macro F1</div>
            <div class="metric-card-sub">Balanced average across classes</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">89.6%</div>
            <div class="metric-card-lbl">Avg Unknown Detection</div>
            <div class="metric-card-sub">Leave-one-out (95%ile threshold)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-val">4.2%</div>
            <div class="metric-card-lbl">Known False Alarm</div>
            <div class="metric-card-sub">Controlled to ~5% validation target</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section A: Known Device Classification
    st.markdown("### A. Known Device Classification Performance")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 2.6])

    with ctrl_col1:
        chosen_split = st.radio(
            "Evaluation Split:",
            options=["Test Split (1,067 windows)", "Validation Split (1,071 windows)"],
            index=0,
            help="Switch between final single-touch test split and intermediate validation tuning split.",
        )
        is_test_split = "Test" in chosen_split

    with ctrl_col2:
        chart_metric = st.selectbox(
            "Primary Metric to Graph:",
            options=["F1-Score", "Accuracy / Support", "Precision", "Recall"],
            index=0,
        )

    with ctrl_col3:
        selected_devices = st.multiselect(
            "Filter Devices to Display:",
            options=DEVICE_NAMES,
            default=DEVICE_NAMES,
            format_func=lambda x: f"{DEVICE_ICONS.get(x, '')} {DEVICE_FRIENDLY.get(x, x)}",
        )

    if is_test_split:
        current_per_class = ref_per_class_test
    else:
        current_per_class = {
            "AmazonEcho": {"precision": 0.9950, "recall": 0.9659, "f1": 0.9802, "support": 205},
            "BelkinWemoMotionSensor": {"precision": 0.9714, "recall": 0.9951, "f1": 0.9831, "support": 409},
            "PhilipsHue": {"precision": 0.9837, "recall": 0.9757, "f1": 0.9797, "support": 247},
            "SamsungCamera": {"precision": 0.9949, "recall": 0.9849, "f1": 0.9899, "support": 199},
            "TPLinkSmartPlug": {"precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "support": 11},
        }

    # Clean DataFrame for Altair
    rows_data = []
    for d in selected_devices:
        stats = current_per_class.get(d, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
        rows_data.append({
            "device": d,
            "device_name": DEVICE_FRIENDLY.get(d, d),
            "icon": DEVICE_ICONS.get(d, ""),
            "precision": float(stats["precision"] * 100),
            "recall": float(stats["recall"] * 100),
            "f1_score": float(stats["f1"] * 100),
            "window_count": int(stats["support"]),
        })

    df_display = pd.DataFrame(rows_data)

    if not df_display.empty:
        chart_col, table_col = st.columns([1.6, 1.4])

        with chart_col:
            st.markdown(f"**Device Comparison — {chart_metric}**")
            
            if chart_metric == "F1-Score":
                metric_field = "f1_score"
                y_title = "F1-Score (%)"
                is_pct = True
            elif chart_metric == "Precision":
                metric_field = "precision"
                y_title = "Precision (%)"
                is_pct = True
            elif chart_metric == "Recall":
                metric_field = "recall"
                y_title = "Recall (%)"
                is_pct = True
            else:
                metric_field = "window_count"
                y_title = "Window Count (Support)"
                is_pct = False

            zoom_y = False
            if is_pct:
                zoom_y = st.checkbox("🔍 Zoom Y-axis (80% – 100%)", value=False, help="Focus in on the top 20% to observe fine differences between devices.")
                if zoom_y:
                    y_scale = alt.Scale(zero=False, domain=[80, 102])
                else:
                    y_scale = alt.Scale(domain=[0, 105])
            else:
                y_scale = alt.Scale(zero=True)

            chart = (
                alt.Chart(df_display)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("device_name:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=220)),
                    y=alt.Y(f"{metric_field}:Q", title=y_title, scale=y_scale),
                    color=alt.Color(
                        "device:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=DEVICE_NAMES,
                            range=["#38bdf8", "#818cf8", "#f472b6", "#34d399", "#fbbf24"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("device_name:N", title="Device"),
                        alt.Tooltip("f1_score:Q", title="F1-Score (%)", format=".2f"),
                        alt.Tooltip("precision:Q", title="Precision (%)", format=".2f"),
                        alt.Tooltip("recall:Q", title="Recall (%)", format=".2f"),
                        alt.Tooltip("window_count:Q", title="Window Count", format="d"),
                    ],
                )
                .properties(height=360)
            )

            text = chart.mark_text(
                align="center",
                baseline="bottom",
                dy=-5,
                color="#e2e8f0",
                fontSize=12,
                fontWeight=600,
            ).encode(
                text=alt.Text(f"{metric_field}:Q", format=".1f" if is_pct else "d")
            )

            st.altair_chart(chart + text, use_container_width=True)

        with table_col:
            st.markdown("**Per-Device Detailed Metrics**")
            df_table = df_display[["device_name", "f1_score", "precision", "recall", "window_count"]].copy()
            df_table.columns = ["Device", "F1-Score (%)", "Precision (%)", "Recall (%)", "Windows (Support)"]
            st.dataframe(df_table, use_container_width=True, hide_index=True)

            if "TPLinkSmartPlug" in selected_devices:
                st.info(
                    "💡 **Note on TP-Link Smart Plug:** Because this smart plug transmits packets in rare ~4-minute "
                    "idle intervals, its total window count is small (9 in test, 11 in val). While precision/recall "
                    "reach 100%, each single window has a large statistical weight."
                )
    else:
        st.warning("Please select at least one device from the filter above.")

    # Section B: Open-Set / Unknown Device Detection Benchmark
    st.markdown("---")
    st.markdown("### B. Open-Set / Unknown Device Detection Benchmark")
    st.markdown(
        "To rigorously test unknown detection without assuming it works, we performed a **leave-one-device-out cross-validation**. "
        "Each device was removed from training in turn to simulate a truly unseen transmitter. Thresholds were selected on the "
        "**known validation split** and evaluated on the **unseen test split**:"
    )

    thresh_choice = st.radio(
        "Validation Threshold Budget:",
        options=["95th Percentile Threshold (~5% False Alarm Budget)", "99th Percentile Threshold (~1% False Alarm Budget)"],
        horizontal=True,
    )
    is_95th = "95th" in thresh_choice

    if openset_data and "leave_one_out_benchmarks" in openset_data:
        loo_dict = openset_data["leave_one_out_benchmarks"]
        loo_rows = []
        for dev, vals in loo_dict.items():
            if is_95th:
                tpr = vals["unknown_tpr_95"]
                fpr = vals["known_fpr_95"]
                min_c = vals["min_conf_95"] * 100
            else:
                tpr = vals["unknown_tpr_99"]
                fpr = vals["known_fpr_99"]
                min_c = vals["min_conf_99"] * 100

            loo_rows.append({
                "Held-Out Device": DEVICE_FRIENDLY.get(dev, dev),
                "device_key": dev,
                "Unknown Windows": vals["test_unknown_windows"],
                "Unknown Detection Rate (TPR %)": tpr,
                "Known False Alarm Rate (FPR %)": fpr,
                "Min Confidence Required (%)": min_c,
            })

        df_loo = pd.DataFrame(loo_rows)

        col_loo_chart, col_loo_table = st.columns([1.6, 1.4])

        with col_loo_chart:
            st.markdown(f"**Unknown Detection Rate (TPR) by Held-Out Device**")
            chart_loo = (
                alt.Chart(df_loo)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Held-Out Device:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=220)),
                    y=alt.Y("Unknown Detection Rate (TPR %):Q", title="Detection Rate (%)", scale=alt.Scale(domain=[0, 105])),
                    color=alt.condition(
                        alt.datum["Unknown Detection Rate (TPR %)"] > 80,
                        alt.value("#10b981"),
                        alt.value("#f59e0b"),
                    ),
                    tooltip=[
                        alt.Tooltip("Held-Out Device:N", title="Device"),
                        alt.Tooltip("Unknown Detection Rate (TPR %):Q", title="Detection Rate (%)", format=".1f"),
                        alt.Tooltip("Known False Alarm Rate (FPR %):Q", title="Known FPR (%)", format=".1f"),
                        alt.Tooltip("Unknown Windows:Q", title="Test Windows", format="d"),
                    ],
                )
                .properties(height=340)
            )

            text_loo = chart_loo.mark_text(
                align="center",
                baseline="bottom",
                dy=-5,
                color="#e2e8f0",
                fontSize=12,
                fontWeight=600,
            ).encode(text=alt.Text("Unknown Detection Rate (TPR %):Q", format=".1f"))

            st.altair_chart(chart_loo + text_loo, use_container_width=True)

        with col_loo_table:
            st.markdown("**Leave-One-Device-Out Benchmark Summary**")
            st.dataframe(
                df_loo.drop(columns=["device_key"]),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(f"""
            - **Average Detection Rate (TPR):** `{df_loo['Unknown Detection Rate (TPR %)'].mean():.1f}%`
            - **Average Known False Alarm (FPR):** `{df_loo['Known False Alarm Rate (FPR %)'].mean():.1f}%` (strictly controlled on test set)
            """)

        st.warning(
            "**Key Open-Set Finding & Limitations:**\n"
            "- Devices with distinct timing profiles (`TP-Link Smart Plug`, `Samsung SmartCam`) are flagged with **100.0% reliability**.\n"
            "- Devices with overlapping burst characteristics (`Belkin WeMo Motion Sensor`) reach **66.2%** at a 5% false alarm budget, "
            "dropping to **14.2%** at a strict 1% budget. Open-set detection purely from timing metadata is fundamentally bounded by timing overlap."
        )

    # Feature Importance & Confusion Matrix
    st.markdown("---")
    cm_col, fi_col = st.columns([1.3, 1.7])

    with cm_col:
        st.subheader("Confusion Matrix (Saved)")
        cm_file = "cm_raw_test.png" if is_test_split else "cm_raw_val.png"
        cm_path = os.path.join(RESULTS_DIR, cm_file)
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix ({'Test' if is_test_split else 'Validation'} Split)", use_container_width=True)
        else:
            st.info(f"Confusion matrix plot ({cm_file}) not found in results directory.")

    with fi_col:
        st.subheader("Top Timing Indicators")
        st.markdown("Features with highest Gini importance in the Random Forest model:")
        
        feature_names = [
            "raw_mean", "raw_std", "raw_min", "raw_max", "raw_median",
            "raw_skew", "raw_kurtosis", "raw_p25", "raw_p75", "raw_p95",
            "raw_p99", "raw_energy", "raw_entropy"
        ]
        
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": rf_model.feature_importances_,
            }).sort_values(by="Importance", ascending=True)

            fi_chart = (
                alt.Chart(fi_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#6366f1")
                .encode(
                    x=alt.X("Importance:Q", title="Relative Feature Importance"),
                    y=alt.Y("Feature:N", sort="-x", title=None),
                    tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
                )
                .properties(height=380)
            )
            st.altair_chart(fi_chart, use_container_width=True)
            st.caption("Percentile 75 (`raw_p75`), `raw_median`, and `raw_p25` provide the most discriminative timing signal.")
        else:
            st.info("Feature importance data is unavailable without loading the trained model.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: TRY AN EXAMPLE (INTERACTIVE TEST WINDOW INFERENCE)
# ═════════════════════════════════════════════════════════════════════════════
with tab_example:
    st.subheader("Try a Saved Test Window")
    st.markdown(
        "Experience the Random Forest classifier and open-set unknown detection in action! "
        "Select any sample from the **unseen test split** (`data/features/raw_scaled_test.csv`) "
        "to observe how the timing indicators lead to the predicted device or trigger an **Unknown Device** flag."
    )
    st.caption("🔒 All examples here are authentic 128-packet timing windows captured during the November 24, 2016 test period.")

    if df_test is not None and rf_model is not None:
        ctrl_c1, ctrl_c2 = st.columns([1.5, 1.5])
        with ctrl_c1:
            enable_openset = st.checkbox(
                "🛡️ Enable Open-Set Verification (Flag Unknown if Confidence < Threshold)",
                value=True,
                help="When enabled, windows with prediction confidence below the threshold are flagged as 'Unknown Device' rather than forced into a known class.",
            )

        with ctrl_c2:
            min_conf_thresh = st.slider(
                "Minimum Confidence Required (%):",
                min_value=50,
                max_value=99,
                value=84,
                disabled=not enable_openset,
                help="Calibrated on the validation split: 84.0% corresponds to the 95th percentile validation anomaly threshold (allowing ~5% false alarms on known traffic).",
            )

        sel_c1, sel_c2, sel_c3 = st.columns([1.5, 2.0, 1.2])

        with sel_c1:
            dev_filter = st.selectbox(
                "Filter Test Windows / Action:",
                options=["All Devices"] + DEVICE_NAMES + ["SIMULATE_UNKNOWN"],
                format_func=lambda x: (
                    "All Devices (1,067)" if x == "All Devices"
                    else "🚨 Simulate New / Unknown Rogue Device" if x == "SIMULATE_UNKNOWN"
                    else f"{DEVICE_ICONS.get(x, '')} {DEVICE_FRIENDLY.get(x, x)}"
                ),
            )

        is_simulated_unknown = (dev_filter == "SIMULATE_UNKNOWN")

        if is_simulated_unknown:
            with sel_c2:
                st.info("🧪 **Live Simulation Active:** Testing an unverified smart device with an anomalous timing rhythm.")
            with sel_c3:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.button("🔄 Reset to Known", on_click=lambda: st.session_state.update({"chosen_idx": 0}))

            true_device = "UnknownDevice"
            window_id = 9999
            time_block_id = 99
            features_vector = np.array([0.5, 1.5, -0.5, 2.0, 0.2, 0.8, 1.0, 0.0, 0.6, 1.2, 1.8, 1.0, 0.5]).reshape(1, -1)
            selected_row = {col: features_vector[0][i] for i, col in enumerate(FEATURE_COLS)}
            selected_row["device_label"] = "UnknownDevice"
            selected_row["window_id"] = 9999
            selected_row["time_block_id"] = 99
        else:
            if dev_filter == "All Devices":
                filtered_df = df_test
            else:
                filtered_df = df_test[df_test["device_label"] == dev_filter]

            with sel_c2:
                available_indices = filtered_df.index.tolist()
                sample_count = len(available_indices)
                
                if "chosen_idx" not in st.session_state:
                    st.session_state["chosen_idx"] = available_indices[0] if available_indices else 0

                if st.session_state["chosen_idx"] not in available_indices and available_indices:
                    st.session_state["chosen_idx"] = available_indices[0]

                chosen_index = st.select_slider(
                    f"Choose Window (1 of {sample_count} available):",
                    options=available_indices,
                    value=st.session_state["chosen_idx"],
                    format_func=lambda x: f"Window #{df_test.loc[x, 'window_id']} ({df_test.loc[x, 'device_label']})",
                )
                st.session_state["chosen_idx"] = chosen_index

            with sel_c3:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🎲 Random Sample", use_container_width=True):
                    st.session_state["chosen_idx"] = random.choice(available_indices)
                    st.rerun()

            # Extract selected row
            selected_row = df_test.loc[st.session_state["chosen_idx"]]
            true_device = selected_row["device_label"]
            window_id = int(selected_row["window_id"])
            time_block_id = int(selected_row["time_block_id"])
            features_vector = np.array([float(selected_row[col]) for col in FEATURE_COLS]).reshape(1, -1)

        # Model Inference
        pred_device = rf_model.predict(features_vector)[0]
        pred_probs = rf_model.predict_proba(features_vector)[0]
        classes = list(rf_model.classes_)

        max_prob = pred_probs[classes.index(pred_device)]
        confidence = max_prob * 100
        anomaly_score = 1.0 - max_prob

        # Decision with open-set verification
        is_unknown = enable_openset and (confidence < min_conf_thresh)

        if is_unknown:
            final_decision = "Unknown Device"
            box_class = "pred-unknown"
            status_icon = "⚠️ FLAGGED AS UNKNOWN"
            display_pred = "Unknown Device (Untrusted Timing Profile)"
            pred_color = "#f59e0b"
            firewall_action = "🛑 QUARANTINE / BLOCK FROM PRIVATE LAN"
            firewall_color = "#f43f5e"
            firewall_sub = "Redirected to Isolated Guest VLAN (Policy Triggered)"
        else:
            final_decision = pred_device
            is_correct = (true_device == pred_device)
            box_class = "pred-match" if is_correct else "pred-mismatch"
            status_icon = "✅ VERIFIED KNOWN" if is_correct else "❌ MISCLASSIFICATION"
            display_pred = f"{DEVICE_ICONS.get(pred_device, '')} {DEVICE_FRIENDLY.get(pred_device, pred_device)}"
            pred_color = "#34d399" if is_correct else "#f87171"
            firewall_action = "✅ ALLOW TRAFFIC TO PRIVATE LAN"
            firewall_color = "#34d399"
            firewall_sub = "Device Fingerprint Matches Trusted Hardware Inventory"

        st.markdown(f"""
        <div class="pred-box {box_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span class="chip chip-blue">Test Window #{window_id}</span>
                    <span class="chip chip-amber">Time Block #{time_block_id}</span>
                    <span class="chip {'chip-amber' if is_unknown else 'chip-green'}">{status_icon}</span>
                </div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #f8fafc;">
                    Classification Confidence: {confidence:.1f}% &middot; Anomaly Score: {anomaly_score:.3f}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.2rem; margin-top: 1.2rem;">
                <div style="background: rgba(15,23,42,0.5); padding: 1rem; border-radius: 8px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Ground Truth (Actual Hardware)</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin-top: 0.2rem;">
                        {'🚨 Unseen / Rogue Device' if is_simulated_unknown else f"{DEVICE_ICONS.get(true_device, '')} {DEVICE_FRIENDLY.get(true_device, true_device)}"}
                    </div>
                </div>
                <div style="background: rgba(15,23,42,0.5); padding: 1rem; border-radius: 8px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">System ML Decision</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {pred_color}; margin-top: 0.2rem;">
                        {display_pred}
                    </div>
                </div>
                <div style="background: rgba(15,23,42,0.5); padding: 1rem; border-radius: 8px;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Router Firewall Action</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: {firewall_color}; margin-top: 0.2rem;">
                        {firewall_action}
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem;">{firewall_sub}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if is_unknown:
            st.warning(
                f"🛡️ **Open-Set Flag Triggered:** Confidence ({confidence:.1f}%) is below the required {min_conf_thresh}% threshold. "
                f"Even though the raw Random Forest favored `{pred_device}`, the system rejected it as an **Unknown Device** and triggered the router quarantine policy."
            )

        # Probabilities & Waveform Breakdown
        res_col1, res_col2 = st.columns([1.5, 1.5])

        with res_col1:
            st.markdown("**Posterior Probability Distribution Across Classes:**")
            prob_df = pd.DataFrame({
                "device_name": [DEVICE_FRIENDLY.get(c, c) for c in classes],
                "raw_device": classes,
                "probability": pred_probs * 100,
            }).sort_values(by="probability", ascending=False)

            prob_chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                .encode(
                    x=alt.X("probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("device_name:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.raw_device == pred_device,
                        alt.value("#10b981" if not is_unknown else "#f59e0b"),
                        alt.value("#64748b"),
                    ),
                    tooltip=[alt.Tooltip("device_name:N", title="Device"), alt.Tooltip("probability:Q", title="Probability (%)", format=".2f")],
                )
                .properties(height=220)
            )
            st.altair_chart(prob_chart, use_container_width=True)

        with res_col2:
            st.markdown(f"**Timing Waveform (128 Consecutive Packets):**")
            if is_simulated_unknown:
                np.random.seed(42)
                iat_sequence = np.abs(np.sin(np.linspace(0, 8, 128)) * 1.5 + np.random.normal(0, 0.25, 128))
            else:
                iat_sequence = load_window_sequence(true_device, window_id)

            if iat_sequence is not None:
                wave_df = pd.DataFrame({
                    "packet_index": np.arange(len(iat_sequence)),
                    "iat_log1p": iat_sequence,
                })
                wave_chart = (
                    alt.Chart(wave_df)
                    .mark_line(color="#38bdf8", strokeWidth=2)
                    .encode(
                        x=alt.X("packet_index:Q", title="Packet Index in Window (0 to 127)"),
                        y=alt.Y("iat_log1p:Q", title="log1p(IAT) (seconds)"),
                        tooltip=[alt.Tooltip("packet_index:Q", title="Packet Index"), alt.Tooltip("iat_log1p:Q", title="log1p(IAT)", format=".4f")],
                    )
                    .properties(height=220)
                )
                st.altair_chart(wave_chart, use_container_width=True)
                st.caption(f"Visual rhythm of {DEVICE_FRIENDLY.get(true_device, true_device)} during Window #{window_id}.")
            else:
                st.info("Raw sequence data for this window was not located on disk.")

        with st.expander("🔎 View the 13 Extracted Statistical Features for this Window"):
            feat_display_rows = []
            for col in FEATURE_COLS:
                scaled_val = float(selected_row[col])
                feat_display_rows.append({
                    "Feature Name": col,
                    "Normalized Value (Z-Score)": f"{scaled_val:.4f}",
                    "Role": (
                        "Median IAT scale" if col == "raw_median" else
                        "75th percentile gap" if col == "raw_p75" else
                        "Shortest burst gap" if col == "raw_min" else
                        "Longest idle gap" if col == "raw_max" else
                        "Timing tail heaviness" if col in ("raw_kurtosis", "raw_skew") else
                        "Signal unpredictability" if col == "raw_entropy" else "General timing moment"
                    ),
                })
            st.dataframe(pd.DataFrame(feat_display_rows), use_container_width=True, hide_index=True)

    else:
        st.warning("Test dataset (`raw_scaled_test.csv`) or model (`rf_raw_model.pkl`) not found in the workspace.")


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# TAB: UNKNOWN DEVICE ANALYSIS (OPEN-SET EVALUATION)
# ═════════════════════════════════════════════════════════════════════════════
with tab_unknown:
    st.subheader("Unknown Device Analysis")
    st.markdown(
        "Evaluate network traffic from devices not present in the 5-class training inventory. "
        "Traffic is processed through the same pipeline (128-packet windows, log1p IAT, 13 statistical timing features, Z-score scaling) "
        "and classified by the trained Random Forest model."
    )

    # Compute validation-derived threshold dynamically from data
    if df_val is not None and rf_model is not None:
        val_probs_all = rf_model.predict_proba(df_val[FEATURE_COLS])
        val_confs_all = np.max(val_probs_all, axis=1)
        conf_threshold = round(float(np.percentile(val_confs_all, 5)), 4)
    else:
        conf_threshold = 0.8400

    # Controls Row
    st.markdown("---")
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2.5, 1.5, 1.0])

    # Build device options
    unknown_options = []
    if df_unk_features is not None:
        counts = df_unk_features["device_label"].value_counts().to_dict()
        for dev in sorted(counts.keys()):
            unknown_options.append(f"{dev} ({counts[dev]} windows)")

    control_options = [f"Control: {dev} (Test Set)" for dev in DEVICE_NAMES]
    all_options = unknown_options + control_options

    with ctrl_c1:
        selected_option = st.selectbox(
            "Select Device from UNSW Dataset:",
            options=all_options,
            index=0 if all_options else None,
            help="Select an unseen device from the UNSW Sydney benchmark dataset or a known control device from the test set."
        )

    with ctrl_c2:
        st.markdown(f"**Rejection Threshold:** `{conf_threshold:.4f}`")
        st.caption("5th percentile of max class probability on 1,071 known validation windows (5% false alarm budget).")

    with ctrl_c3:
        st.write("")
        st.write("")
        run_btn = st.button("Run Pipeline", use_container_width=True)

    if selected_option and rf_model is not None and scaler_dict is not None:
        # Determine if selected is control or unknown
        is_control = selected_option.startswith("Control:")
        if is_control:
            dev_name = selected_option.replace("Control: ", "").replace(" (Test Set)", "")
            sub_df = df_test[df_test["device_label"] == dev_name].copy() if df_test is not None else pd.DataFrame()
            X_eval = sub_df[FEATURE_COLS].values
            dev_mac = "Known Control"
        else:
            dev_name = selected_option.split(" (")[0]
            sub_df = df_unk_features[df_unk_features["device_label"] == dev_name].copy()
            dev_mac = sub_df["device_mac"].iloc[0] if "device_mac" in sub_df.columns else "Unknown"
            # Apply identical training scaler: (X - mean) / std
            mean_vec = scaler_dict["mean"]
            std_vec = scaler_dict["std"]
            feat_cols = scaler_dict["feature_names"]
            X_eval = (sub_df[feat_cols].values - mean_vec) / std_vec

        if len(X_eval) > 0:
            classes = list(rf_model.classes_)
            preds = rf_model.predict(X_eval)
            probs = rf_model.predict_proba(X_eval)
            confidences = np.max(probs, axis=1)
            entropies = -np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0)), axis=1)

            total_windows = len(preds)
            mean_conf = float(np.mean(confidences))
            mean_entropy = float(np.mean(entropies))
            dominant_class = pd.Series(preds).mode()[0]
            dominant_share = float(np.mean(preds == dominant_class) * 100)
            low_conf_mask = confidences < conf_threshold
            low_conf_count = int(np.sum(low_conf_mask))
            low_conf_pct = float(low_conf_count / total_windows * 100)

            # Verdict Line
            st.markdown("---")
            if mean_conf >= conf_threshold:
                verdict_text = f"Matches known device: {dominant_class}"
                verdict_color = "#334155"
            else:
                verdict_text = "Low confidence: possibly unfamiliar device"
                verdict_color = "#b91c1c"

            st.markdown(
                f"<div style='border: 1px solid #cbd5e1; background-color: #f8fafc; padding: 14px 18px; border-radius: 4px; margin-bottom: 16px;'>"
                f"<div style='font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase;'>Classification Verdict</div>"
                f"<div style='font-size: 1.25rem; font-weight: 700; color: {verdict_color}; margin-top: 4px;'>{verdict_text}</div>"
                f"<div style='font-size: 0.85rem; color: #475569; margin-top: 4px;'>"
                f"Mean Confidence: <strong>{mean_conf:.4f}</strong> &middot; Threshold: <strong>{conf_threshold:.4f}</strong> &middot; "
                f"Dominant Label: <strong>{dominant_class}</strong> ({dominant_share:.1f}% of windows) &middot; "
                f"Flagged Low Confidence: <strong>{low_conf_count} / {total_windows}</strong> ({low_conf_pct:.1f}%)"
                f"</div></div>",
                unsafe_allow_html=True
            )

            # Summary Metrics Table
            metrics_summary_df = pd.DataFrame([{
                "Device Identifier": f"{dev_name} ({dev_mac})",
                "Total Windows (128 pkts)": total_windows,
                "Dominant Label": dominant_class,
                "Dominant Share": f"{dominant_share:.1f}%",
                "Mean Confidence": f"{mean_conf:.4f}",
                "Mean Entropy (bits)": f"{mean_entropy:.3f}",
                "Flagged Low Confidence (< 0.8400)": f"{low_conf_count} / {total_windows} ({low_conf_pct:.1f}%)",
            }])
            st.dataframe(metrics_summary_df, use_container_width=True, hide_index=True)

            # Class Probability Distribution (Average across device windows)
            st.markdown("**Average Class Probability Distribution Across Windows**")
            mean_probs = np.mean(probs, axis=0)
            prob_dist_df = pd.DataFrame({
                "Trained Class": classes,
                "Average Assigned Probability": [round(float(p), 4) for p in mean_probs],
                "Windows Dominating": [int(np.sum(preds == c)) for c in classes],
                "Share of Windows (%)": [round(float(np.mean(preds == c) * 100), 2) for c in classes],
            })
            st.dataframe(prob_dist_df, use_container_width=True, hide_index=True)

            # Clean Charts: Predictions and Confidence Distribution
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                df_pred_counts = pd.DataFrame({
                    "Predicted Class": classes,
                    "Window Count": [int(np.sum(preds == c)) for c in classes],
                })
                chart_pred = alt.Chart(df_pred_counts).mark_bar(color="#475569").encode(
                    x=alt.X("Predicted Class:N", sort="-y", title="Predicted Class"),
                    y=alt.Y("Window Count:Q", title="Number of Windows"),
                    tooltip=["Predicted Class", "Window Count"]
                ).properties(title="Window Distribution by Predicted Class", height=240)
                st.altair_chart(chart_pred, use_container_width=True)

            with col_chart2:
                df_confs = pd.DataFrame({"Confidence": confidences})
                chart_hist = alt.Chart(df_confs).mark_bar(color="#64748b").encode(
                    x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=20), title="Prediction Confidence (Max Probability)", scale=alt.Scale(domain=[0.0, 1.0])),
                    y=alt.Y("count():Q", title="Window Count"),
                    tooltip=[alt.Tooltip("count()", title="Windows")]
                ).properties(title="Confidence Distribution (Dashed Line = Threshold)", height=240)

                rule = alt.Chart(pd.DataFrame({"x": [conf_threshold]})).mark_rule(
                    color="#b91c1c", strokeDash=[4, 4], size=1.5
                ).encode(x="x:Q")

                st.altair_chart(chart_hist + rule, use_container_width=True)

            # Individual Window Table (first 50 windows)
            with st.expander(f"Inspect Per-Window Model Outputs (First {min(50, total_windows)} of {total_windows} Windows)"):
                preview_rows = []
                for idx in range(min(50, total_windows)):
                    w_dict = {
                        "Window Index": idx,
                        "Predicted Label": preds[idx],
                        "Max Confidence": f"{confidences[idx]:.4f}",
                        "Entropy (bits)": f"{entropies[idx]:.3f}",
                        "Status": "Low Confidence (< 0.8400)" if confidences[idx] < conf_threshold else "Above Threshold",
                    }
                    for ci, cname in enumerate(classes):
                        w_dict[f"P({cname})"] = f"{probs[idx][ci]:.4f}"
                    preview_rows.append(w_dict)
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        else:
            st.info("No windows available for the selected device.")

    # Honest Note on Closed-Set Classifier Limitations
    st.markdown("---")
    st.markdown("**Closed-Set Classifier Architecture and Detection Limitations**")
    st.markdown(
        "The model is a closed-set Random Forest classifier trained on 5 specific IoT devices. "
        "Because output probabilities are normalized to sum to 1.0, the model is architecturally forced to assign 100% of its probability "
        "mass across the 5 known classes, even when presented with network traffic from an unseen hardware model.\n\n"
        "The confidence threshold (`0.8400`) operates as an empirical post-hoc heuristic: it flags windows whose probability distribution "
        "is split or diffuse. However, when an unknown device shares similar timing cadence with a known device "
        "(for example, the Belkin WeMo Switch uses identical UPnP discovery intervals as the Belkin WeMo Motion Sensor, and the Awair monitor\'s "
        "periodic telemetry bursts resemble the Amazon Echo idle loop), the model assigns high confidence to the incorrect class. "
        "Consequently, the threshold is an anomaly screening filter rather than an infallible guarantee."
    )

    # Benchmark Evaluation Across All Devices
    st.markdown("---")
    st.markdown("### Benchmark Evaluation: All Unknown Devices vs Known Controls")
    st.markdown(
        f"All unknown devices and known test sets evaluated under the identical pipeline with the validation-calibrated threshold (`{conf_threshold:.4f}`)."
    )

    if df_unk_eval is not None:
        st.dataframe(df_unk_eval, use_container_width=True, hide_index=True)
    elif os.path.exists(os.path.join(PROJ_ROOT, "results", "unknown_evaluation_table.csv")):
        df_tbl = pd.read_csv(os.path.join(PROJ_ROOT, "results", "unknown_evaluation_table.csv"))
        st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    # Calibration Performance Numbers
    st.markdown(
        "<div style=\'border-top: 1px solid #e2e8f0; padding-top: 12px; margin-top: 12px; font-size: 0.9rem; color: #334155;\'>"
        "<strong>Empirical Calibration & False Alarm Performance:</strong><br>"
        "&bull; <strong>Calibration Threshold:</strong> 0.8400 (5th percentile of max class probability on 1,071 known validation windows)<br>"
        "&bull; <strong>Known Test False Alarm Rate:</strong> 3.94% (42 / 1,067 known test windows incorrectly flagged below 0.8400)<br>"
        "&bull; <strong>Unknown Device Flagging Rate:</strong> 67.58% (3,370 / 4,987 total unknown windows correctly flagged as low confidence)<br>"
        "&bull; <strong>High-Rejection Devices (>= 90% flagged):</strong> Nest DropCam (100%), Netatmo Weather (100%), PixStar Photo Frame (100%), iHome (100%), "
        "Withings Smart Scale (100%), Nest Protect (100%), Triby Speaker (99.6%), LiFX Bulb (98.8%), HP Printer (94.8%), Netatmo Welcome (92.8%), Canary Camera (92.2%)<br>"
        "&bull; <strong>Low-Rejection Devices (< 50% flagged due to timing overlap):</strong> Samsung SmartThings (63.4%), Belkin WeMo Switch (30.8% - mislabeled Belkin Motion), "
        "Awair Air Quality (1.6% - mislabeled Amazon Echo), Withings Sleep Sensor (0.6% - mislabeled Amazon Echo)"
        "</div>",
        unsafe_allow_html=True
    )

    # Devices with Insufficient Packets (Skipped)
    st.markdown("---")
    st.markdown("### Excluded Candidate Devices (Insufficient Packets in Capture Window)")
    st.markdown(
        "Candidate devices from the UNSW Sydney benchmark dataset that had zero packet activity during the Nov 22–24, 2016 capture period "
        "and were therefore omitted from evaluation:"
    )
    df_skipped = pd.DataFrame([
        {"Device Name": "August Doorbell Cam", "MAC Address": "e0:76:d0:33:bb:85", "Observed Packets": 0, "Windows Formed": 0, "Reason": "Device offline during capture window"},
        {"Device Name": "Belkin NetCam", "MAC Address": "00:0e:8f:b7:47:32", "Observed Packets": 0, "Windows Formed": 0, "Reason": "Device offline during capture window"},
        {"Device Name": "Blipcare Blood Pressure Meter", "MAC Address": "74:6a:89:12:f5:4c", "Observed Packets": 0, "Windows Formed": 0, "Reason": "Device offline during capture window"},
        {"Device Name": "Hello Barbie", "MAC Address": "e0:76:d0:3f:00:ae", "Observed Packets": 0, "Windows Formed": 0, "Reason": "Device offline during capture window"},
        {"Device Name": "Ring Doorbell", "MAC Address": "40:f3:08:ff:1e:da", "Observed Packets": 0, "Windows Formed": 0, "Reason": "Device offline during capture window"},
        {"Device Name": "TP-Link Day Night Cloud Camera", "MAC Address": "ec:1a:59:79:f4:89", "Observed Packets": 0, "Windows Formed": 0, "Reason": "Device offline during capture window"},
        {"Device Name": "Withings Smart Baby Monitor", "MAC Address": "00:24:e4:20:28:c6", "Observed Packets": 0, "Windows Formed": 0, "Reason": "Device offline during capture window"},
    ])
    st.dataframe(df_skipped, use_container_width=True, hide_index=True)


# TAB 5: STUDENT PRESENTATION CHEAT SHEET & FAQ
# ═════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.subheader("🎓 Student Presentation Cheat Sheet")
    st.markdown(
        "Key questions, design decisions, and defensive rationale to help present this project confidently:"
    )

    q1, q2 = st.columns(2)

    with q1:
        with st.container():
            st.markdown("#### 1. Why use Inter-Arrival Time (IAT) instead of packet contents?")
            st.markdown(
                "Modern IoT protocols use end-to-end encryption (TLS, DTLS, AES). Inspecting payloads is "
                "impossible without breaking encryption or violating user privacy. Packet sizes can also be easily padded. "
                "However, **inter-arrival timing is governed by hardware crystals, firmware task loops, and physical sensor triggers**, "
                "which creates a unique, indelible behavioral fingerprint."
            )

        with st.container():
            st.markdown("#### 2. Why the `log1p(IAT)` transform instead of P99 percentile clipping?")
            st.markdown(
                "Calculating a 99th percentile threshold across the dataset before splitting leaks future test data "
                "into preprocessing (violating fundamental data hygiene). In contrast, `log(1 + IAT)` is a **fixed mathematical "
                "function with zero data-derived parameters**, compressing microseconds to minutes without data leakage."
            )

        with st.container():
            st.markdown("#### 3. How does the system flag Unknown Devices?")
            st.markdown(
                "We compute the prediction anomaly score: $S(x) = 1.0 - \\max_c P(y = c \\mid x)$. "
                "If the Random Forest's top confidence does not meet the threshold calibrated on the **known validation split**, "
                "the window is flagged as **`Unknown Device`**. This prevents forced false-positive classifications on unseen hardware."
            )

    with q2:
        with st.container():
            st.markdown("#### 4. Why contiguous time-block splitting rather than random shuffle?")
            st.markdown(
                "A random train/test split shuffles packets from the same minute across train and test. "
                "Correlated events (like an active motion detection burst) would be seen in both splits, giving an artificially inflated accuracy. "
                "By grouping windows into **1-hour blocks and assigning contiguous blocks (70% early train, 15% mid val, 15% late test)**, "
                "the model is tested strictly against future temporal intervals."
            )

        with st.container():
            st.markdown("#### 5. Why did earlier VAE experiments struggle on unknown detection?")
            st.markdown(
                "Reconstruction-error VAEs model feature density blindly without task supervision. "
                "When an unseen device shares similar feature magnitudes (e.g. Belkin WeMo motion spikes overlapping with Amazon Echo activity), "
                "the VAE reconstructs it easily, yielding low reconstruction error and poor anomaly detection (3–14%). "
                "In contrast, **discriminative confidence thresholding directly measures whether traffic falls outside the learned decision boundaries**, "
                "achieving **89.6%** average unknown detection across all held-out devices."
            )

        with st.container():
            st.markdown("#### 6. Summary of Key Numbers to Mention")
            st.markdown(
                "- **Closed-Set Test Accuracy:** 99.34% (Macro F1: 99.38%)\n"
                "- **Open-Set Unknown Detection (TPR @ 95%ile):** 89.6% average (100% on TP-Link and Samsung, 93.1% Echo, 88.4% Hue, 66.2% Belkin)\n"
                "- **Known False Alarm Rate (FPR @ 95%ile):** 4.2% on test split (controlled to ~5% validation target)\n"
                "- **Dataset:** 973,974 packets across 5 devices over 3 days (Nov 22–24, 2016)\n"
                "- **Windows:** 7,606 non-overlapping windows of 128 packets\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "IoT Device Fingerprinting Project &middot; Computer Networks &middot; Built with Python & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
