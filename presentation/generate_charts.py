"""
Generate high-resolution dark-themed charts for presentation slides.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Shared styling constants
BG_COLOR = "#0F172A"       # Deep slate navy
CARD_BG = "#1E293B"        # Card background
ACCENT_CYAN = "#38BDF8"    # Sky cyan
ACCENT_BLUE = "#60A5FA"    # Light blue
ACCENT_GREEN = "#34D399"   # Emerald
ACCENT_AMBER = "#FBBF24"   # Amber
ACCENT_ROSE = "#F43F5E"    # Rose
TEXT_LIGHT = "#F8FAFC"     # White
TEXT_MUTED = "#94A3B8"     # Muted slate
BORDER_COLOR = "#334155"   # Slate border

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "text.color": TEXT_LIGHT,
    "axes.labelcolor": TEXT_LIGHT,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.edgecolor": BORDER_COLOR,
    "figure.facecolor": CARD_BG,
    "axes.facecolor": CARD_BG,
})

def create_timing_intuition_chart():
    """Slide 4: Intuition - Packet arrival timeline for 3 device behaviors."""
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 4.2), sharex=True, facecolor=CARD_BG)
    fig.subplots_adjust(hspace=0.45, left=0.22, right=0.96, top=0.88, bottom=0.18)

    np.random.seed(42)

    # 1. Amazon Echo: Bursty voice interactions
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    t_echo = np.concatenate([
        np.sort(np.random.uniform(0.5, 2.0, 35)),
        np.sort(np.random.uniform(5.0, 6.5, 30)),
        np.sort(np.random.uniform(8.0, 9.5, 35))
    ])
    ax1.vlines(t_echo, 0, 1, color=ACCENT_CYAN, lw=1.8, alpha=0.9)
    ax1.set_ylabel("Echo (Audio)", fontsize=11, fontweight="bold", color=TEXT_LIGHT)
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.text(0.6, 1.05, "Voice / Cloud Query Burst", color=ACCENT_CYAN, fontsize=9, fontweight="bold")
    ax1.text(3.0, 0.4, "Idle Silence", color=TEXT_MUTED, fontsize=9, fontstyle="italic")

    # 2. Samsung Camera: Continuous high-frequency video stream
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    t_cam = np.sort(np.random.uniform(0, 10, 220))
    ax2.vlines(t_cam, 0, 1, color=ACCENT_BLUE, lw=1.0, alpha=0.7)
    ax2.set_ylabel("Camera (Stream)", fontsize=11, fontweight="bold", color=TEXT_LIGHT)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.text(0.6, 1.05, "Dense Continuous Video Transmission", color=ACCENT_BLUE, fontsize=9, fontweight="bold")

    # 3. TP-Link Smart Plug: Sparse periodic telemetry / heartbeat
    ax3 = axes[2]
    ax3.set_facecolor(CARD_BG)
    t_plug = np.array([1.2, 4.8, 8.4])
    ax3.vlines(t_plug, 0, 1, color=ACCENT_AMBER, lw=2.5, alpha=0.95)
    ax3.set_ylabel("Plug (Heartbeat)", fontsize=11, fontweight="bold", color=TEXT_LIGHT)
    ax3.set_ylim(0, 1.2)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (seconds)", fontsize=11, color=TEXT_MUTED)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.text(1.3, 1.05, "Periodic Keep-Alive Ping (Zero Payload Read)", color=ACCENT_AMBER, fontsize=9, fontweight="bold")

    out_path = os.path.join(ASSETS_DIR, "timing_intuition.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print("Saved:", out_path)


def create_results_chart():
    """Slide 9: Real classification metrics & feature importances."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=CARD_BG)
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.95, top=0.84, bottom=0.18)

    # Panel 1: Per-Class Test F1-scores
    devices = ["Belkin WeMo", "TP-Link Plug", "Samsung Cam", "Philips Hue", "Amazon Echo"]
    f1_scores = [99.8, 100.0, 99.2, 99.2, 98.8]
    colors = [ACCENT_CYAN, ACCENT_BLUE, ACCENT_CYAN, ACCENT_CYAN, ACCENT_CYAN]

    bars1 = ax1.barh(devices, f1_scores, color=colors, height=0.55, edgecolor="none")
    ax1.set_xlim(95, 101)
    ax1.set_title("Test F1-Score by Device (%)", fontsize=12, fontweight="bold", color=TEXT_LIGHT, pad=12)
    ax1.grid(axis="x", color=BORDER_COLOR, linestyle="--", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_color(BORDER_COLOR)

    for bar, val in zip(bars1, f1_scores):
        ax1.text(val + 0.15, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                 va="center", ha="left", color=TEXT_LIGHT, fontsize=10, fontweight="bold")

    # Panel 2: Top Feature Importances (Random Forest)
    features = ["P75 IAT", "Median IAT", "P25 IAT", "Min IAT", "Kurtosis"]
    importances = [16.7, 13.6, 13.3, 11.3, 11.3]
    colors2 = [ACCENT_CYAN, ACCENT_BLUE, ACCENT_BLUE, ACCENT_AMBER, ACCENT_AMBER]

    bars2 = ax2.barh(features[::-1], importances[::-1], color=colors2[::-1], height=0.55, edgecolor="none")
    ax2.set_xlim(0, 20)
    ax2.set_title("Top Feature Importance (%)", fontsize=12, fontweight="bold", color=TEXT_LIGHT, pad=12)
    ax2.grid(axis="x", color=BORDER_COLOR, linestyle="--", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_color(BORDER_COLOR)

    for bar, val in zip(bars2, importances[::-1]):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                 va="center", ha="left", color=TEXT_LIGHT, fontsize=10, fontweight="bold")

    out_path = os.path.join(ASSETS_DIR, "results_chart.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print("Saved:", out_path)


from matplotlib.patches import FancyBboxPatch

def create_pipeline_diagram():
    """Slide 5: Clean pipeline visual showing 5 stages."""
    fig, ax = plt.subplots(figsize=(10.5, 3.2), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.axis("off")

    stages = [
        ("Raw PCAP\nCapture", "UNSW Nov 2016\n973K Packets"),
        ("Inter-Arrival\nTime (IAT)", "Delta t = t_i - t_{i-1}\nZero Payload Read"),
        ("128-Packet\nWindows", "7,606 Windows\nTemporal Blocks"),
        ("13 Statistical\nFeatures", "Mean, P75, Med,\nMin, Kurtosis..."),
        ("Random\nForest", "100 Estimators\n99.34% Accuracy")
    ]

    n = len(stages)
    box_w = 1.45
    box_h = 1.5
    spacing = 1.95
    start_x = 0.55

    for i, (title, desc) in enumerate(stages):
        x = start_x + i * spacing
        # Draw rounded box
        box = FancyBboxPatch((x, 0.7), box_w, box_h,
                             boxstyle="round,pad=0.12",
                             facecolor=BG_COLOR, edgecolor=ACCENT_CYAN, lw=2.0)
        ax.add_patch(box)
        ax.text(x + box_w/2, 1.75, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=TEXT_LIGHT)
        ax.text(x + box_w/2, 1.1, desc, ha="center", va="center",
                fontsize=8.5, color=TEXT_MUTED)

        # Arrow to next box
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + 0.35, 1.45), xytext=(x + box_w + 0.05, 1.45),
                        arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2.2))

    ax.set_xlim(0, start_x + (n - 1) * spacing + box_w + 0.4)
    ax.set_ylim(0.4, 2.6)

    out_path = os.path.join(ASSETS_DIR, "pipeline_diagram.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print("Saved:", out_path)


def create_future_architecture_diagram():
    """Slide 12: Future unknown device isolation flow."""
    fig, ax = plt.subplots(figsize=(10.5, 3.4), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.axis("off")

    # Ingress Window
    ax.add_patch(FancyBboxPatch((0.4, 0.9), 1.8, 1.3, boxstyle="round,pad=0.12",
                                facecolor=BG_COLOR, edgecolor=BORDER_COLOR, lw=1.8))
    ax.text(1.3, 1.75, "128-Packet\nTraffic Window", ha="center", va="center",
            fontsize=10, fontweight="bold", color=TEXT_LIGHT)
    ax.text(1.3, 1.25, "Extract 13 IAT stats", ha="center", va="center",
            fontsize=8.5, color=TEXT_MUTED)

    # Arrow to RF Model
    ax.annotate("", xy=(2.7, 1.55), xytext=(2.3, 1.55),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2.0))

    # RF Classifier
    ax.add_patch(FancyBboxPatch((2.8, 0.9), 2.0, 1.3, boxstyle="round,pad=0.12",
                                facecolor=BG_COLOR, edgecolor=ACCENT_CYAN, lw=2.0))
    ax.text(3.8, 1.75, "Random Forest\nClassifier", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TEXT_LIGHT)
    ax.text(3.8, 1.25, "P(Device | Timing)", ha="center", va="center",
            fontsize=8.5, color=TEXT_MUTED)

    # Arrow to Decision Diamond/Box
    ax.annotate("", xy=(5.3, 1.55), xytext=(4.9, 1.55),
                arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2.0))

    # Decision Block
    ax.add_patch(FancyBboxPatch((5.4, 0.9), 2.0, 1.3, boxstyle="round,pad=0.12",
                                facecolor=BG_COLOR, edgecolor=ACCENT_AMBER, lw=2.0))
    ax.text(6.4, 1.75, "Confidence Check\n(Validation Calibrated)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=TEXT_LIGHT)
    ax.text(6.4, 1.25, "Anomaly score <= 0.16?\n(Min conf >= 84%)", ha="center", va="center",
            fontsize=8.5, color=ACCENT_AMBER)

    # Top Branch: Known -> Normal LAN
    ax.annotate("", xy=(7.9, 2.0), xytext=(7.5, 1.7),
                arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=2.0))
    ax.add_patch(FancyBboxPatch((8.0, 1.5), 2.3, 1.0, boxstyle="round,pad=0.12",
                                facecolor=BG_COLOR, edgecolor=ACCENT_GREEN, lw=1.8))
    ax.text(9.15, 2.15, "[CURRENT] KNOWN", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=ACCENT_GREEN)
    ax.text(9.15, 1.75, "Assign Device Tag & Allow LAN", ha="center", va="center",
            fontsize=8, color=TEXT_MUTED)

    # Bottom Branch: Unknown -> Firewall / Quarantine
    ax.annotate("", xy=(7.9, 1.1), xytext=(7.5, 1.4),
                arrowprops=dict(arrowstyle="->", color=ACCENT_ROSE, lw=2.0))
    ax.add_patch(FancyBboxPatch((8.0, 0.35), 2.3, 1.0, boxstyle="round,pad=0.12",
                                facecolor=BG_COLOR, edgecolor=ACCENT_ROSE, lw=1.8))
    ax.text(9.15, 1.0, "[PLANNED] UNKNOWN", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=ACCENT_ROSE)
    ax.text(9.15, 0.6, "Quarantine VLAN / Router Drop", ha="center", va="center",
            fontsize=8, color=TEXT_MUTED)

    ax.set_xlim(0, 10.6)
    ax.set_ylim(0.2, 2.7)

    out_path = os.path.join(ASSETS_DIR, "future_architecture.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print("Saved:", out_path)

if __name__ == "__main__":
    create_timing_intuition_chart()
    create_results_chart()
    create_pipeline_diagram()
    create_future_architecture_diagram()
