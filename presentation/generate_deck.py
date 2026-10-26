"""
Generate professional 14-slide widescreen presentation for IoT Fingerprinting project.
Adheres strictly to design constraints:
- 16:9 widescreen
- Dark navy palette (#0F172A) with sky cyan accent (#38BDF8)
- Strict max ~25 words body text per slide (punchy cards/bullets, not walls of text)
- Detailed plain-English speaker notes on EVERY slide
- High-resolution embedded visual charts
- Saves to presentation/project_overview.pptx and copies to project_overview.pptx
"""
import os
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESENTATION_DIR = os.path.join(PROJ_ROOT, "presentation")
ASSETS_DIR = os.path.join(PRESENTATION_DIR, "assets")
PPTX_PATH = os.path.join(PRESENTATION_DIR, "project_overview.pptx")
ROOT_PPTX_PATH = os.path.join(PROJ_ROOT, "project_overview.pptx")

# Color Palette
COLOR_BG = RGBColor(15, 23, 42)          # Slate 900 (#0F172A)
COLOR_CARD = RGBColor(30, 41, 59)        # Slate 800 (#1E293B)
COLOR_CARD_BORDER = RGBColor(51, 65, 85) # Slate 700 (#334155)
COLOR_ACCENT = RGBColor(56, 189, 248)    # Sky 400 (#38BDF8)
COLOR_ACCENT_GREEN = RGBColor(52, 211, 153) # Emerald 400 (#34D399)
COLOR_ACCENT_AMBER = RGBColor(251, 191, 36) # Amber 400 (#FBBF24)
COLOR_ACCENT_ROSE = RGBColor(244, 63, 94)   # Rose 500 (#F43F5E)
COLOR_TEXT = RGBColor(248, 250, 252)     # Slate 50 (#F8FAFC)
COLOR_MUTED = RGBColor(148, 163, 184)    # Slate 400 (#94A3B8)

FONT_HEADING = "Helvetica"
FONT_BODY = "Arial"

def init_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs

def set_slide_background(slide):
    """Draw a dark navy background rectangle filling the entire slide."""
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5)
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = COLOR_BG
    bg_shape.line.fill.background()
    return bg_shape

def add_header(slide, category, title, subtitle=None):
    """Add a consistent, elegant header to a slide."""
    header_box = slide.shapes.add_textbox(Inches(0.9), Inches(0.55), Inches(11.5), Inches(1.3))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Category Pill
    p_cat = tf.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = FONT_HEADING
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ACCENT
    p_cat.space_after = Pt(4)

    # Title
    p_title = tf.add_paragraph()
    p_title.text = title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(32)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT
    
    if subtitle:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(15)
        p_sub.font.color.rgb = COLOR_MUTED
        p_sub.space_before = Pt(3)

def add_card(slide, left, top, width, height, title, lines, title_color=COLOR_ACCENT, bg_color=COLOR_CARD, border_color=COLOR_CARD_BORDER):
    """Add a structured card with rounded corners and clean typography."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.5)

    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.32)
    tf.margin_right = Inches(0.32)
    tf.margin_top = Inches(0.32)
    tf.margin_bottom = Inches(0.32)

    if title:
        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.name = FONT_HEADING
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = title_color
        p_title.space_after = Pt(12)

    for i, line in enumerate(lines):
        p = tf.add_paragraph() if (title or i > 0) else tf.paragraphs[0]
        p.text = line
        p.font.name = FONT_BODY
        p.font.size = Pt(16)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(8)

    return card

def set_speaker_notes(slide, notes):
    """Set detailed plain-English speaker notes."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes.strip()

# ==============================================================================
# Slide Builders (Strictly <= 25 body words per slide)
# ==============================================================================

def build_slide_1(prs):
    """Slide 1: Title Slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)

    tb = slide.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.9), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True

    p_badge = tf.paragraphs[0]
    p_badge.text = "IOT NETWORK SECURITY & MACHINE LEARNING"
    p_badge.font.name = FONT_HEADING
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = COLOR_ACCENT
    p_badge.space_after = Pt(14)

    p_title = tf.add_paragraph()
    p_title.text = "Encrypted IoT Device Fingerprinting"
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(40)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT
    p_title.space_after = Pt(12)

    p_sub = tf.add_paragraph()
    p_sub.text = "Passive hardware identification via inter-packet timing rhythms"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(20)
    p_sub.font.color.rgb = COLOR_MUTED
    p_sub.space_after = Pt(28)

    p_meta = tf.add_paragraph()
    p_meta.text = "5 Smart Devices  •  99.34% Accuracy  •  Zero Payload Decryption"
    p_meta.font.name = FONT_HEADING
    p_meta.font.size = Pt(16)
    p_meta.font.bold = True
    p_meta.font.color.rgb = COLOR_ACCENT

    set_speaker_notes(slide, """
Welcome everyone. Today I'm presenting our project on Encrypted IoT Device Fingerprinting.
In modern smart homes, dozens of connected devices operate simultaneously—from smart speakers to security cameras. While nearly all of their traffic is encrypted to safeguard user privacy, the physical hardware, operating systems, and firmware of these devices transmit packets according to distinct, repeatable timing rhythms.
In this work, we demonstrate how a gateway can accurately recognize five distinct IoT devices with over ninety-nine percent accuracy using purely inter-packet arrival times—without decrypting, inspecting, or storing a single byte of application payload.
""")
    return slide

def build_slide_2(prs):
    """Slide 2: The Problem."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Problem & Motivation", "The Smart Home Blind Spot", "Why traditional network security fails on modern consumer IoT")

    col_w = Inches(3.6)
    gap = Inches(0.4)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "Encrypted Payloads",
             ["TLS/HTTPS encryption blinds deep packet inspection tools."],
             title_color=COLOR_ACCENT)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "Spoofed Headers",
             ["MAC and IP addresses are trivially faked in software."],
             title_color=COLOR_ACCENT_AMBER)

    add_card(slide, start_x + (col_w + gap)*2, top_y, col_w, h,
             "Zero Visibility",
             ["Gateways cannot verify what physical hardware is connected."],
             title_color=COLOR_ACCENT_ROSE)

    set_speaker_notes(slide, """
Why is IoT device identification so difficult?
First, modern privacy standards encrypt traffic end-to-end. Traditional firewalls and intrusion detection tools that rely on Deep Packet Inspection are completely blind.
Second, network-layer identifiers like MAC and IP addresses are easily spoofed by malicious software.
Finally, consumers rarely patch or configure smart appliances, leaving home gateways with zero visibility into what hardware is actually running on their local network.
We need a passive, physical-layer-adjacent method to verify identity without breaking encryption.
""")
    return slide

def build_slide_3(prs):
    """Slide 3: Core Principle."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Technical Foundation", "Timing Leaks Hardware Identity", "How encrypted streams reveal their physical source")

    top_y = Inches(2.4)
    w = Inches(11.5)
    start_x = Inches(0.9)

    add_card(slide, start_x, top_y, w, Inches(2.0),
             "Hardware Rhythms",
             ["Microcontrollers, polling loops, and firmware emit distinct packet delays."],
             title_color=COLOR_ACCENT)

    add_card(slide, start_x, top_y + Inches(2.3), w, Inches(2.0),
             "The Footsteps Analogy",
             ["Identifying devices by packet timing is like recognizing footsteps in the dark."],
             title_color=COLOR_ACCENT_GREEN)

    set_speaker_notes(slide, """
The foundational premise of our project is that timing leaks identity.
Even when payloads are strongly encrypted and randomized, the underlying hardware cannot hide when it transmits.
A camera encoding H.264 video streams packets in dense bursts driven by frame rates. A smart plug queries local sensors on a fixed microcontroller clock cycle. A smart speaker bursts network traffic only when audio processing buffers fill.
Think of it like recognizing someone walking down a hallway purely by the rhythm and cadence of their footsteps, even in total darkness.
""")
    return slide

def build_slide_4(prs):
    """Slide 4: Timing Intuition (Visual)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Behavioral Profiles", "How Devices Differ in Time", "Direct comparison of real packet arrival timelines")

    chart_path = os.path.join(ASSETS_DIR, "timing_intuition.png")
    if os.path.exists(chart_path):
        slide.shapes.add_picture(chart_path, Inches(0.9), Inches(2.2), width=Inches(11.5))

    add_card(slide, Inches(0.9), Inches(6.55), Inches(11.5), Inches(0.65),
             "",
             ["Hardware tasks emit distinct packet rhythms across idle, streaming, and query states."],
             bg_color=COLOR_CARD, border_color=COLOR_CARD_BORDER)

    set_speaker_notes(slide, """
Here we see real packet timeline traces illustrating the stark differences between device profiles.
Notice the top trace: an Amazon Echo sits completely quiet until voice interaction triggers a dense burst of cloud requests.
In the middle trace, a Samsung security camera streams continuous, dense packets to maintain real-time video feeds.
At the bottom, a TP-Link smart plug operates on a barebones microcontroller, emitting sparse, periodic heartbeat pings every few minutes.
These timing rhythms are physically coupled to hardware function and cannot be concealed without adding deliberate, latency-inducing traffic padding.
""")
    return slide

def build_slide_5(prs):
    """Slide 5: The End-to-End Pipeline."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "System Architecture", "End-to-End Traffic Pipeline", "A lightweight, passive 5-stage inference flow")

    chart_path = os.path.join(ASSETS_DIR, "pipeline_diagram.png")
    if os.path.exists(chart_path):
        slide.shapes.add_picture(chart_path, Inches(0.9), Inches(2.2), width=Inches(11.5))

    add_card(slide, Inches(0.9), Inches(5.5), Inches(11.5), Inches(1.3),
             "Passive Gateway Processing",
             ["Fast user-space pipeline extracting summary statistics with zero decryption overhead."],
             title_color=COLOR_ACCENT)

    set_speaker_notes(slide, """
Here is our complete processing pipeline. It consists of five streamlined steps.
First, we capture raw network traffic from the gateway interface.
Second, we compute Inter-Arrival Time deltas between successive packets for each IP stream.
Third, we segment these deltas into non-overlapping windows of 128 packets.
Fourth, we calculate thirteen lightweight summary statistics per window.
Finally, a Random Forest classifier evaluates the feature vector and outputs the device identity.
The entire feature extraction requires simple arithmetic operations that can run natively on consumer router hardware.
""")
    return slide

def build_slide_6(prs):
    """Slide 6: Dataset & The 5 Devices."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Dataset Specification", "UNSW Real-World IoT Dataset", "Empirical benchmark traffic captured over 3 continuous days")

    col_w = Inches(3.6)
    gap = Inches(0.4)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "973K Packets",
             ["Captured across 3 continuous days in a live smart home."],
             title_color=COLOR_ACCENT)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "7,606 Windows",
             ["Segmented into non-overlapping 128-packet temporal blocks."],
             title_color=COLOR_ACCENT_GREEN)

    add_card(slide, start_x + (col_w + gap)*2, top_y, col_w, h,
             "5 Devices",
             ["Amazon Echo, Belkin WeMo, Philips Hue, Samsung Camera, TP-Link Plug."],
             title_color=COLOR_ACCENT_AMBER)

    set_speaker_notes(slide, """
Our empirical evaluation is grounded in the benchmark UNSW Smart Home dataset, collected over three continuous days in a live testbed environment.
The capture contains nearly one million raw packets across five popular consumer devices: the Amazon Echo smart speaker, the Belkin WeMo motion sensor, the Philips Hue smart lighting bridge, a Samsung smart security camera, and a TP-Link smart plug.
Segmenting into 128-packet chunks yielded 7,606 feature windows, giving us a robust distribution across both high-bandwidth streaming and low-power telemetry devices.
""")
    return slide

def build_slide_7(prs):
    """Slide 7: Feature Engineering (13 Timing Moments)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Feature Engineering", "13 Statistical Timing Moments", "Compact mathematical summaries of packet inter-arrival times (IAT)")

    col_w = Inches(3.6)
    gap = Inches(0.4)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "Spread & Center",
             ["Mean, Median, Min, Max, and Standard Deviation."],
             title_color=COLOR_ACCENT)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "Clock Quantiles",
             ["25th, 75th, 95th, and 99th percentiles capture hardware timers."],
             title_color=COLOR_ACCENT_GREEN)

    add_card(slide, start_x + (col_w + gap)*2, top_y, col_w, h,
             "Shape & Energy",
             ["Skewness, Kurtosis, Signal Energy, and Shannon Entropy."],
             title_color=COLOR_ACCENT_AMBER)

    set_speaker_notes(slide, """
Rather than using complex wavelets or heavy deep learning that would struggle on embedded routers, we extract thirteen statistical timing moments for each window.
These include standard measures like mean, standard deviation, min, and max to capture baseline bandwidth and idle gaps.
More importantly, percentiles—specifically the 25th, 75th, and median—capture the fixed clock cycles of device microcontrollers.
Kurtosis and skewness capture sudden traffic spikes versus steady streaming. Finally, signal energy and entropy quantify burst irregularity without inspecting a single byte of payload.
""")
    return slide

def build_slide_8(prs):
    """Slide 8: Model Architecture & Evaluation Rigor."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Methodology & Evaluation", "Model Architecture & Rigor", "Preventing temporal data leakage through block-level splitting")

    col_w = Inches(3.6)
    gap = Inches(0.4)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "Random Forest",
             ["100 trees provide sub-millisecond, robust multi-class inference."],
             title_color=COLOR_ACCENT)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "1-Hour Blocks",
             ["70% Train, 15% Validation, and 15% Held-Out Test."],
             title_color=COLOR_ACCENT_GREEN)

    add_card(slide, start_x + (col_w + gap)*2, top_y, col_w, h,
             "Method Rigor",
             ["Zero temporal leakage; test split evaluated exactly once."],
             title_color=COLOR_ACCENT_AMBER)

    set_speaker_notes(slide, """
For classification, we use a Random Forest ensemble with 100 decision trees.
But our primary focus was methodological rigor. Standard random cross-validation leaks temporal patterns because packets from the same session appear in both training and testing.
To avoid this fatal flaw, we implemented contiguous one-hour block splitting: the first 70% of chronological time blocks formed our training set (5,468 windows), the next 15% formed our validation set (1,071 windows), and the final 15% formed our held-out test set (1,067 windows).
The test set was touched exactly once after all hyperparameters were locked.
""")
    return slide

def build_slide_9(prs):
    """Slide 9: Results & Performance."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Experimental Results", "99.34% Classification Accuracy", "Empirical performance across 1,067 held-out test windows")

    chart_path = os.path.join(ASSETS_DIR, "results_chart.png")
    if os.path.exists(chart_path):
        slide.shapes.add_picture(chart_path, Inches(0.9), Inches(2.2), width=Inches(11.5))

    add_card(slide, Inches(0.9), Inches(6.55), Inches(11.5), Inches(0.65),
             "",
             ["99.34% Test Accuracy  •  99.38% Macro F1  •  P75 and Median drive 30.3% of splits."],
             bg_color=COLOR_CARD, border_color=COLOR_ACCENT)

    set_speaker_notes(slide, """
Here are our final test results on the held-out temporal partition.
The model achieved 99.34% overall test accuracy and a 99.38% macro F1-score across all 1,067 test windows.
Looking at the per-device chart on the left, four of the five devices scored over 99% F1, with Belkin WeMo at 99.8% and TP-Link Plug at 100%.
On the right, our feature importance analysis reveals that the 75th percentile and median IAT account for over 30% of decision tree splits.
This confirms our hypothesis: fixed microcontroller polling intervals provide the strongest fingerprint for device identification.
""")
    return slide

def build_slide_10(prs):
    """Slide 10: Capabilities & Limitations."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "System Analysis", "Capabilities & Constraints", "Real-world trade-offs in passive traffic fingerprinting")

    col_w = Inches(5.5)
    gap = Inches(0.5)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "System Strengths",
             ["• Zero payload decryption preserves user privacy.",
              "• Sub-millisecond router inference resists spoofing."],
             title_color=COLOR_ACCENT_GREEN)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "Real-World Limitations",
             ["• Severe Wi-Fi jitter distorts packet delays.",
              "• Identical hardware models share timing signatures."],
             title_color=COLOR_ACCENT_AMBER)

    set_speaker_notes(slide, """
Let's honestly assess what the current system can and cannot do.
On the capability side, it operates entirely without payload decryption, preserving end-to-end user privacy. Feature extraction and tree inference take less than a millisecond per window, easily fitting within router CPU and memory budgets.
However, real-world constraints exist: extreme Wi-Fi congestion or latency jitter can distort inter-arrival intervals.
Furthermore, two identical hardware models running the same firmware emit identical fingerprints.
Finally, standard classifiers assume a closed world, classifying every unseen packet as one of the known five.
""")
    return slide

def build_slide_11(prs):
    """Slide 11: Project Status & Roadmap."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Project Scope", "Project Status & Roadmap", "Clear distinction between implemented work and planned future work")

    col_w = Inches(5.5)
    gap = Inches(0.5)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "[DONE] Implemented & Verified",
             ["• 13 IAT statistical feature pipeline.",
              "• 5-device Random Forest classifier (99.34% accuracy).",
              "• Interactive Streamlit demonstration dashboard."],
             title_color=COLOR_ACCENT_GREEN)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "[PLANNED] Future Milestones",
             ["• Open-set unknown rejection via anomaly thresholds.",
              "• Automated router firewall isolation (iptables).",
              "• Dynamic quarantine VLAN for unverified hardware."],
             title_color=COLOR_ACCENT_AMBER)

    set_speaker_notes(slide, """
To be crystal clear about project status:
The completed, fully validated portion is our 13-feature timing extractor and 5-device Random Forest classifier, along with our interactive presentation dashboard.
What remains as planned future work is two-fold: first, robust open-set unknown device detection to reliably reject rogue hardware, and second, automated router-level enforcement using Linux iptables or OpenWrt firewall rules to quarantine unverified devices onto an isolated VLAN.
""")
    return slide

def build_slide_12(prs):
    """Slide 12: Future Architecture (Unknown Device Isolation)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Future Architecture", "Planned: Gateway Unknown Isolation", "Automated anomaly thresholding and router-level quarantine")

    chart_path = os.path.join(ASSETS_DIR, "future_architecture.png")
    if os.path.exists(chart_path):
        slide.shapes.add_picture(chart_path, Inches(0.9), Inches(2.2), width=Inches(11.5))

    add_card(slide, Inches(0.9), Inches(5.5), Inches(11.5), Inches(1.3),
             "Decision Thresholding",
             ["Confidence >= 84% allows verified LAN access; lower confidence routes to quarantine."],
             title_color=COLOR_ACCENT)

    set_speaker_notes(slide, """
This diagram outlines our planned gateway interceptor architecture.
When an active device transmits 128 packets, the gateway extracts its timing fingerprint and queries the Random Forest.
We calibrated an anomaly threshold on the validation split: if tree agreement falls below 84%, the traffic is flagged as an unknown anomaly.
In preliminary leave-one-out experiments, this threshold successfully flagged 89.6% of unseen device windows while maintaining a 4% false alarm rate on known devices.
Future router integrations will automatically redirect flagged traffic to a captive quarantine VLAN.
""")
    return slide

def build_slide_13(prs):
    """Slide 13: Privacy & Security Implications."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Security & Ethics", "Privacy & Ethical Considerations", "Balancing network defense against behavioral side-channels")

    col_w = Inches(5.5)
    gap = Inches(0.5)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "Defensive Privacy Benefit",
             ["• Secures local networks without decrypting private user data.",
              "• Automatically inventories connected smart appliances."],
             title_color=COLOR_ACCENT_GREEN)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "Surveillance Risk (Dual-Use)",
             ["• Timing side-channels can leak occupant presence and sleep.",
              "• Mitigations require device-level random traffic padding."],
             title_color=COLOR_ACCENT_ROSE)

    set_speaker_notes(slide, """
Whenever we discuss traffic fingerprinting, privacy and ethical implications must be front and center.
From a defensive perspective, this approach is privacy-preserving because network administrators can inventory smart devices without decrypting sensitive personal messages, health data, or video streams.
However, the exact same timing side-channels present a surveillance risk: an external eavesdropper passively monitoring home Wi-Fi can infer when occupants wake up, leave home, or sleep, purely from device traffic bursts.
Defending against such side-channel leaks may require deliberate traffic shaping at the device level.
""")
    return slide

def build_slide_14(prs):
    """Slide 14: Key Takeaways & Conclusion."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide)
    add_header(slide, "Summary & Outlook", "Key Takeaways & Conclusion", "Summary of contributions and next steps")

    col_w = Inches(3.6)
    gap = Inches(0.4)
    start_x = Inches(0.9)
    top_y = Inches(2.3)
    h = Inches(4.3)

    add_card(slide, start_x, top_y, col_w, h,
             "Timing Works",
             ["99.34% accuracy identifying 5 devices without decrypting payloads."],
             title_color=COLOR_ACCENT)

    add_card(slide, start_x + col_w + gap, top_y, col_w, h,
             "Clocks Matter",
             ["Median and 75th percentile IAT capture physical polling hardware."],
             title_color=COLOR_ACCENT_GREEN)

    add_card(slide, start_x + (col_w + gap)*2, top_y, col_w, h,
             "Next Frontier",
             ["Harden open-set unknown rejection and implement router blocking."],
             title_color=COLOR_ACCENT_AMBER)

    set_speaker_notes(slide, """
To conclude our presentation:
First, packet timing is a remarkably robust fingerprint for IoT hardware, achieving 99.34% accuracy without touching encrypted payloads.
Second, statistical quantiles like median and 75th percentile IAT capture hardware clock cycles most effectively.
Third, our next steps will focus on moving beyond the closed-set model to harden unknown device rejection and implement automated gateway isolation.
Thank you for your time, and I am now happy to open the floor to any questions.
""")
    return slide

# ==============================================================================
# Main Generation
# ==============================================================================

def main():
    prs = init_presentation()

    print("Building slides...")
    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)
    build_slide_4(prs)
    build_slide_5(prs)
    build_slide_6(prs)
    build_slide_7(prs)
    build_slide_8(prs)
    build_slide_9(prs)
    build_slide_10(prs)
    build_slide_11(prs)
    build_slide_12(prs)
    build_slide_13(prs)
    build_slide_14(prs)

    prs.save(PPTX_PATH)
    print(f"Saved deck to {PPTX_PATH}")

    # Copy to root
    shutil.copyfile(PPTX_PATH, ROOT_PPTX_PATH)
    print(f"Copied deck to {ROOT_PPTX_PATH}")

if __name__ == "__main__":
    main()
