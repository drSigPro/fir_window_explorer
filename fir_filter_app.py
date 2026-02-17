"""
FIR Filter Design using the Windowing Method – Interactive Streamlit App
=========================================================================
Author : Mahesh Panicker (mahesh.signalproc@gmail.com) aided by Google Antigravity
Purpose: Demonstrate FIR filter design via the window method for UG students.
"""

import streamlit as st
import numpy as np
from scipy import signal
from scipy.special import i0 as bessel_i0
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIR Filter Design – Window Method",
    page_icon="📐",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    .math-box {
        background: #f8f9fc;
        border-left: 4px solid #667eea;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 18px;
        margin: 8px 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">FIR Filter Design – Window Method</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  Helper: Ideal impulse responses
# ──────────────────────────────────────────────────────────────

def ideal_lowpass(N: int, wc: float) -> np.ndarray:
    """Ideal lowpass impulse response centred at n = (N-1)/2."""
    n = np.arange(N)
    mid = (N - 1) / 2.0
    d = n - mid
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.sin(wc * d) / (np.pi * d)
    h[d == 0] = wc / np.pi
    return h


def ideal_highpass(N: int, wc: float) -> np.ndarray:
    n = np.arange(N)
    mid = (N - 1) / 2.0
    h_lp = ideal_lowpass(N, wc)
    h = -h_lp.copy()
    h[n == mid] = 1.0 - wc / np.pi
    return h


def ideal_bandpass(N: int, wc1: float, wc2: float) -> np.ndarray:
    return ideal_lowpass(N, wc2) - ideal_lowpass(N, wc1)


def ideal_bandstop(N: int, wc1: float, wc2: float) -> np.ndarray:
    n = np.arange(N)
    mid = (N - 1) / 2.0
    h_bp = ideal_bandpass(N, wc1, wc2)
    h = -h_bp.copy()
    h[n == mid] = 1.0 - (wc2 - wc1) / np.pi
    return h


# ──────────────────────────────────────────────────────────────
#  Helper: Window functions (manual, for displaying formulas)
# ──────────────────────────────────────────────────────────────

WINDOW_NAMES = ["Rectangular", "Hann", "Hamming", "Blackman", "Kaiser"]

WINDOW_FORMULAS = {
    "Rectangular": r"w[n] = 1, \quad 0 \le n \le N{-}1",
    "Hann":        r"w[n] = 0.5 - 0.5\cos\!\left(\frac{2\pi n}{N{-}1}\right)",
    "Hamming":     r"w[n] = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{N{-}1}\right)",
    "Blackman":    r"w[n] = 0.42 - 0.5\cos\!\left(\frac{2\pi n}{N{-}1}\right) + 0.08\cos\!\left(\frac{4\pi n}{N{-}1}\right)",
    "Kaiser":      r"w[n] = \frac{I_0\!\left(\beta\sqrt{1 - \left(\frac{2n}{N{-}1} - 1\right)^2}\right)}{I_0(\beta)}",
}

WINDOW_PROPERTIES = {
    # (approx transition width factor, stopband atten dB, passband ripple dB)
    "Rectangular": (0.91, 21, -0.9),
    "Hann":        (3.32, 44, -0.06),
    "Hamming":     (3.44, 55, -0.02),  # using 55 as given in reference
    "Blackman":    (5.98, 75, -0.0014),
    "Kaiser":      (None, None, None),  # depends on beta
}


def get_window(name: str, N: int, beta: float = 6.0) -> np.ndarray:
    """Return window coefficients."""
    if name == "Rectangular":
        return np.ones(N)
    elif name == "Hann":
        return signal.windows.hann(N, sym=True)
    elif name == "Hamming":
        return signal.windows.hamming(N, sym=True)
    elif name == "Blackman":
        return signal.windows.blackman(N, sym=True)
    elif name == "Kaiser":
        return signal.windows.kaiser(N, beta, sym=True)
    else:
        return np.ones(N)


# ──────────────────────────────────────────────────────────────
#  Helper: Frequency response
# ──────────────────────────────────────────────────────────────

def freq_response(h: np.ndarray, n_fft: int = 2048):
    """Return (omega, H) where omega ∈ [0, π]."""
    w, H = signal.freqz(h, worN=n_fft, whole=False)
    return w, H


# ──────────────────────────────────────────────────────────────
#  Plotly helpers
# ──────────────────────────────────────────────────────────────

COLORS = {
    "Rectangular": "#e74c3c",
    "Hann":        "#2ecc71",
    "Hamming":     "#3498db",
    "Blackman":    "#9b59b6",
    "Kaiser":      "#e67e22",
}


def _plotly_layout(title, xaxis_title, yaxis_title, height=400):
    return dict(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Design Parameters")

    filter_type = st.selectbox("**Filter Type**", ["Lowpass", "Highpass", "Bandpass", "Bandstop"])

    st.markdown("---")

    if filter_type in ("Lowpass", "Highpass"):
        wc = st.slider(
            "**Cutoff frequency ωc (× π rad/sample)**",
            min_value=0.01, max_value=0.99, value=0.4, step=0.01,
            help="Normalized cutoff frequency. Multiply by π to get rad/sample."
        )
        wc_rad = wc * np.pi
        wc2_rad = None
    else:
        col1, col2 = st.columns(2)
        with col1:
            wc1 = st.slider("**ωc₁ (× π)**", 0.01, 0.98, 0.3, 0.01)
        with col2:
            wc2 = st.slider("**ωc₂ (× π)**", 0.02, 0.99, 0.6, 0.01)
        if wc2 <= wc1:
            st.error("ωc₂ must be greater than ωc₁")
            st.stop()
        wc_rad = wc1 * np.pi
        wc2_rad = wc2 * np.pi

    st.markdown("---")

    N = st.slider("**Filter order N**", min_value=5, max_value=127, value=31, step=2,
                  help="Number of taps. Must be odd for a symmetric (Type I) FIR filter.")
    if N % 2 == 0:
        N += 1
        st.info(f"Adjusted to N = {N} (odd) for a Type I FIR filter.")

    st.markdown("---")

    window_choice = st.selectbox("**Window Function**", WINDOW_NAMES)

    beta = 6.0
    if window_choice == "Kaiser":
        beta = st.slider("**Kaiser β**", 0.0, 14.0, 6.0, 0.1,
                         help="Shape parameter. Higher β → wider main lobe, lower sidelobes.")

    st.markdown("---")
    st.markdown("##### 📊 Quick Reference")
    st.markdown("""
    | Window | Atten. (dB) | TW factor |
    |--------|-------------|-----------|
    | Rect.  | 21 | 0.91 |
    | Hann   | 44 | 3.32 |
    | Hamming| 55 | 3.44 |
    | Blackman| 75 | 5.98 |
    | Kaiser | tunable | tunable |
    """)

# ──────────────────────────────────────────────────────────────
#  Compute ideal & windowed filter
# ──────────────────────────────────────────────────────────────

if filter_type == "Lowpass":
    h_ideal = ideal_lowpass(N, wc_rad)
elif filter_type == "Highpass":
    h_ideal = ideal_highpass(N, wc_rad)
elif filter_type == "Bandpass":
    h_ideal = ideal_bandpass(N, wc_rad, wc2_rad)
else:
    h_ideal = ideal_bandstop(N, wc_rad, wc2_rad)

w_win = get_window(window_choice, N, beta)
h_designed = h_ideal * w_win

# ──────────────────────────────────────────────────────────────
#  MAIN TABS
# ──────────────────────────────────────────────────────────────

tab_theory, tab_time, tab_freq, tab_compare, tab_explore = st.tabs([
    "📖 Theory & Math",
    "⏱️ Time Domain",
    "📈 Frequency Response",
    "🔀 Compare Windows",
    "🧪 Design Explorer",
])

# ==================== TAB 1: Theory & Math ====================
with tab_theory:
    st.subheader("Step-by-Step: FIR Design via the Window Method")

    st.markdown("""
    The **window method** designs an FIR filter in four steps:

    1. **Specify the ideal frequency response** $H_d(e^{j\\omega})$
    2. **Compute the ideal impulse response** $h_d[n]$ via the inverse DTFT
    3. **Choose a window** $w[n]$ of length $N$
    4. **Multiply**: $h[n] = h_d[n] \\cdot w[n]$
    """)

    st.markdown("---")

    # Ideal filter formulas
    st.markdown("#### Ideal Impulse Responses")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Lowpass**")
        st.latex(r"h_d[n] = \frac{\sin(\omega_c (n - \alpha))}{\pi (n - \alpha)}, \quad \alpha = \frac{N-1}{2}")
        st.markdown("**Highpass**")
        st.latex(r"h_d[n] = \delta[n-\alpha] - \frac{\sin(\omega_c (n-\alpha))}{\pi (n-\alpha)}")
    with col_b:
        st.markdown("**Bandpass**")
        st.latex(r"h_d[n] = \frac{\sin(\omega_{c2}(n-\alpha))}{\pi(n-\alpha)} - \frac{\sin(\omega_{c1}(n-\alpha))}{\pi(n-\alpha)}")
        st.markdown("**Bandstop**")
        st.latex(r"h_d[n] = \delta[n-\alpha] - h_{d,\mathrm{BP}}[n]")

    st.markdown("---")

    # Window formulas
    st.markdown("#### Window Functions")
    for wname in WINDOW_NAMES:
        with st.expander(f"**{wname} Window**", expanded=(wname == window_choice)):
            st.latex(WINDOW_FORMULAS[wname])
            props = WINDOW_PROPERTIES[wname]
            if props[0] is not None:
                st.markdown(f"""
                | Property | Value |
                |----------|-------|
                | Transition width | {props[0]} × fs / N |
                | Stopband attenuation | {props[1]} dB |
                | Passband ripple | {props[2]} dB |
                """)
            else:
                st.markdown("Properties depend on the value of **β**.")
                st.markdown("""
                | β | TW factor | Stopband atten. | Passband ripple |
                |---|-----------|-----------------|-----------------|
                | 6 | 4.33 × fs/N | 64 dB | −0.0057 dB |
                | 8 | 5.25 × fs/N | 81 dB | −0.00087 dB |
                | 10 | 6.36 × fs/N | 100 dB | −0.000013 dB |
                """)

    st.markdown("---")

    st.markdown("#### Design Procedure Summary")
    st.info("""
    1. **From specifications**: Determine the required stopband attenuation and transition width.
    2. **Choose window**: Select the window that meets the attenuation requirement (see table above).
    3. **Estimate N**: $N \\approx \\text{TW\\_factor} \\times f_s / \\Delta f$, where $\\Delta f$ is the transition width in Hz.
    4. **Compute**: $h[n] = h_d[n] \\cdot w[n]$.
    5. **Verify**: Check the frequency response and iterate if necessary.
    """)


# ==================== TAB 2: Time Domain ====================
with tab_time:
    st.subheader("Time-Domain Analysis")

    n_indices = np.arange(N)

    # ---- Window plot ----
    fig_win = go.Figure()
    fig_win.add_trace(go.Scatter(
        x=n_indices, y=w_win, mode="lines+markers",
        name=f"{window_choice} window",
        line=dict(color=COLORS[window_choice], width=2),
        marker=dict(size=5),
    ))
    fig_win.update_layout(**_plotly_layout(
        f"{window_choice} Window  w[n]", "Sample index n", "Amplitude"))
    st.plotly_chart(fig_win)

    # ---- Ideal impulse response ----
    fig_hd = go.Figure()
    fig_hd.add_trace(go.Scatter(
        x=n_indices, y=h_ideal, mode="lines+markers",
        name="h_d[n]  (ideal)",
        line=dict(color="#555", width=2),
        marker=dict(size=5),
    ))
    fig_hd.update_layout(**_plotly_layout(
        f"Ideal {filter_type} Impulse Response  h_d[n]", "Sample index n", "Amplitude"))
    st.plotly_chart(fig_hd)

    # ---- Designed (windowed) impulse response ----
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=n_indices, y=h_designed, mode="lines+markers",
        name="h[n] = h_d[n] · w[n]",
        line=dict(color=COLORS[window_choice], width=2),
        marker=dict(size=5),
    ))
    fig_h.add_trace(go.Scatter(
        x=n_indices, y=h_ideal, mode="lines",
        name="h_d[n] (ideal)",
        line=dict(color="#ccc", width=1, dash="dash"),
    ))
    fig_h.update_layout(**_plotly_layout(
        "Designed Filter  h[n] = h_d[n] · w[n]", "Sample index n", "Amplitude"))
    st.plotly_chart(fig_h)

    # Display the mathematical expression
    st.markdown("#### Current Design Expression")
    if filter_type in ("Lowpass", "Highpass"):
        st.latex(rf"h[n] = h_d[n] \cdot w[n], \quad \omega_c = {wc:.2f}\pi, \quad N = {N}")
    else:
        st.latex(rf"h[n] = h_d[n] \cdot w[n], \quad \omega_{{c1}} = {wc_rad/np.pi:.2f}\pi, \; \omega_{{c2}} = {wc2_rad/np.pi:.2f}\pi, \quad N = {N}")


# ==================== TAB 3: Frequency Response ====================
with tab_freq:
    st.subheader("Frequency Response")

    w_axis, H = freq_response(h_designed)
    H_mag = np.abs(H)
    H_db = 20 * np.log10(np.maximum(H_mag, 1e-12))
    H_phase = np.unwrap(np.angle(H))

    # Magnitude (linear)
    fig_mag = go.Figure()
    fig_mag.add_trace(go.Scatter(
        x=w_axis / np.pi, y=H_mag, mode="lines",
        line=dict(color=COLORS[window_choice], width=2),
        name="|H(e^jω)|",
    ))
    # Mark cutoff(s)
    if filter_type in ("Lowpass", "Highpass"):
        fig_mag.add_vline(x=wc, line_dash="dash", line_color="gray",
                          annotation_text=f"ωc = {wc:.2f}π")
    else:
        fig_mag.add_vline(x=wc_rad / np.pi, line_dash="dash", line_color="gray",
                          annotation_text=f"ωc₁ = {wc_rad/np.pi:.2f}π")
        fig_mag.add_vline(x=wc2_rad / np.pi, line_dash="dash", line_color="gray",
                          annotation_text=f"ωc₂ = {wc2_rad/np.pi:.2f}π")
    fig_mag.update_layout(**_plotly_layout(
        "Magnitude Response  |H(e^jω)|", "Normalized Frequency (× π rad/sample)", "Magnitude"))
    st.plotly_chart(fig_mag)

    # Magnitude (dB)
    fig_db = go.Figure()
    fig_db.add_trace(go.Scatter(
        x=w_axis / np.pi, y=H_db, mode="lines",
        line=dict(color=COLORS[window_choice], width=2),
        name="20 log₁₀|H|",
    ))
    if filter_type in ("Lowpass", "Highpass"):
        fig_db.add_vline(x=wc, line_dash="dash", line_color="gray",
                         annotation_text=f"ωc = {wc:.2f}π")
    else:
        fig_db.add_vline(x=wc_rad / np.pi, line_dash="dash", line_color="gray",
                         annotation_text=f"ωc₁")
        fig_db.add_vline(x=wc2_rad / np.pi, line_dash="dash", line_color="gray",
                         annotation_text=f"ωc₂")
    fig_db.update_layout(**_plotly_layout(
        "Magnitude Response (dB)", "Normalized Frequency (× π rad/sample)", "Magnitude (dB)"))
    fig_db.update_yaxes(range=[max(-120, np.min(H_db) - 10), 5])
    st.plotly_chart(fig_db)

    # Phase
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(
        x=w_axis / np.pi, y=H_phase, mode="lines",
        line=dict(color=COLORS[window_choice], width=2),
        name="∠H(e^jω)",
    ))
    fig_ph.update_layout(**_plotly_layout(
        "Phase Response  ∠H(e^jω)", "Normalized Frequency (× π rad/sample)", "Phase (radians)"))
    st.plotly_chart(fig_ph)

    # Key metrics
    st.markdown("#### Key Metrics")
    peak_sidelobe = np.max(H_db[H_db < H_db.max()]) if len(H_db) > 1 else 0
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Peak magnitude | {np.max(H_mag):.4f} |
    | Min stopband (dB) | {np.min(H_db):.1f} dB |
    | Filter length | {N} taps |
    """)


# ==================== TAB 4: Compare Windows ====================
with tab_compare:
    st.subheader("Compare All Windows")
    st.markdown("All windows applied to the **same ideal filter** with the current settings.")

    compare_beta = st.slider("Kaiser β for comparison", 0.0, 14.0, 6.0, 0.5, key="compare_beta")

    # Build all filters
    all_filters = {}
    for wname in WINDOW_NAMES:
        b = compare_beta if wname == "Kaiser" else 6.0
        w_tmp = get_window(wname, N, b)
        h_tmp = h_ideal * w_tmp
        all_filters[wname] = {"window": w_tmp, "h": h_tmp}

    # ---- Window comparison ----
    fig_wc = go.Figure()
    for wname in WINDOW_NAMES:
        fig_wc.add_trace(go.Scatter(
            x=n_indices, y=all_filters[wname]["window"],
            mode="lines", name=wname,
            line=dict(color=COLORS[wname], width=2),
        ))
    fig_wc.update_layout(**_plotly_layout(
        "Window Functions Comparison", "Sample index n", "Amplitude", height=380))
    st.plotly_chart(fig_wc)

    # ---- Magnitude (dB) comparison ----
    fig_mc = go.Figure()
    for wname in WINDOW_NAMES:
        w_ax, H_c = freq_response(all_filters[wname]["h"])
        H_c_db = 20 * np.log10(np.maximum(np.abs(H_c), 1e-12))
        fig_mc.add_trace(go.Scatter(
            x=w_ax / np.pi, y=H_c_db,
            mode="lines", name=wname,
            line=dict(color=COLORS[wname], width=2),
        ))
    fig_mc.update_layout(**_plotly_layout(
        "Magnitude Response Comparison (dB)", "Normalized Frequency (× π rad/sample)", "Magnitude (dB)", height=450))
    fig_mc.update_yaxes(range=[-120, 5])
    # Add cutoff markers
    if filter_type in ("Lowpass", "Highpass"):
        fig_mc.add_vline(x=wc, line_dash="dash", line_color="gray", annotation_text=f"ωc")
    else:
        fig_mc.add_vline(x=wc_rad / np.pi, line_dash="dash", line_color="gray", annotation_text="ωc₁")
        fig_mc.add_vline(x=wc2_rad / np.pi, line_dash="dash", line_color="gray", annotation_text="ωc₂")
    st.plotly_chart(fig_mc)

    # ---- Phase comparison ----
    fig_pc = go.Figure()
    for wname in WINDOW_NAMES:
        w_ax, H_c = freq_response(all_filters[wname]["h"])
        fig_pc.add_trace(go.Scatter(
            x=w_ax / np.pi, y=np.unwrap(np.angle(H_c)),
            mode="lines", name=wname,
            line=dict(color=COLORS[wname], width=2),
        ))
    fig_pc.update_layout(**_plotly_layout(
        "Phase Response Comparison", "Normalized Frequency (× π rad/sample)", "Phase (radians)", height=380))
    st.plotly_chart(fig_pc)

    # ---- Summary metrics table ----
    st.markdown("#### Summary Metrics")
    metrics_data = []
    for wname in WINDOW_NAMES:
        w_ax, H_c = freq_response(all_filters[wname]["h"])
        H_c_mag = np.abs(H_c)
        H_c_db = 20 * np.log10(np.maximum(H_c_mag, 1e-12))

        # Estimate main-lobe width from window DFT
        _, W_c = freq_response(all_filters[wname]["window"])
        W_mag = np.abs(W_c)
        W_mag_norm = W_mag / W_mag.max()
        # Main-lobe width ≈ first index where magnitude drops below -3 dB
        below_3db = np.where(20 * np.log10(np.maximum(W_mag_norm, 1e-12)) < -3)[0]
        ml_width = (2 * w_ax[below_3db[0]] / np.pi) if len(below_3db) > 0 else 0

        # Peak sidelobe (in dB relative to main lobe)
        W_db = 20 * np.log10(np.maximum(W_mag_norm, 1e-12))
        # find first null
        first_null_idx = np.argmin(W_mag_norm[:len(W_mag_norm)//2]) if len(W_mag_norm) > 10 else 0
        sidelobe_peak_db = np.max(W_db[first_null_idx:]) if first_null_idx < len(W_db) - 1 else 0

        metrics_data.append({
            "Window": wname,
            "Main-lobe width (× π)": f"{ml_width:.3f}",
            "Peak sidelobe (dB)": f"{sidelobe_peak_db:.1f}",
            "Min stopband (dB)": f"{np.min(H_c_db):.1f}",
        })

    st.table(metrics_data)


# ==================== TAB 5: Design Explorer ====================
with tab_explore:
    st.subheader("🧪 Design Explorer & Learning Aids")

    explore_tab1, explore_tab2, explore_tab3 = st.tabs([
        "📐 Design from Specs",
        "🔬 What-If Explorer",
        "📝 Self-Check Quiz",
    ])

    # ---- Design from specs ----
    with explore_tab1:
        st.markdown("#### Design a filter from specifications")
        st.markdown("Enter your desired specifications and the tool will recommend a window and estimate the filter order.")

        col_spec1, col_spec2 = st.columns(2)
        with col_spec1:
            desired_atten = st.number_input(
                "Desired stopband attenuation (dB)", min_value=10, max_value=120, value=50, step=1)
            transition_width = st.number_input(
                "Transition width Δf (Hz)", min_value=1, max_value=5000, value=500, step=10)
        with col_spec2:
            fs_spec = st.number_input(
                "Sampling frequency fs (Hz)", min_value=100, max_value=100000, value=8000, step=100)

        # Recommend window
        if desired_atten <= 21:
            rec_win, tw_factor = "Rectangular", 0.91
        elif desired_atten <= 44:
            rec_win, tw_factor = "Hann", 3.32
        elif desired_atten <= 55:
            rec_win, tw_factor = "Hamming", 3.44
        elif desired_atten <= 75:
            rec_win, tw_factor = "Blackman", 5.98
        else:
            rec_win = "Kaiser"
            # Estimate Kaiser beta from desired attenuation
            if desired_atten > 50:
                est_beta = 0.1102 * (desired_atten - 8.7)
            elif desired_atten >= 21:
                est_beta = 0.5842 * (desired_atten - 21) ** 0.4 + 0.07886 * (desired_atten - 21)
            else:
                est_beta = 0.0
            tw_factor = (desired_atten - 8) / (2.285 * 2 * np.pi)  # Kaiser approximation
            # Better estimate using Kaiser's formula
            N_kaiser = int(np.ceil((desired_atten - 8) / (2.285 * 2 * np.pi * transition_width / fs_spec))) + 1
            if N_kaiser % 2 == 0:
                N_kaiser += 1

        if rec_win != "Kaiser":
            N_est = int(np.ceil(tw_factor * fs_spec / transition_width))
            if N_est % 2 == 0:
                N_est += 1
        else:
            N_est = N_kaiser

        st.markdown("---")
        st.success(f"""
        **Recommended Design:**
        - **Window**: {rec_win}
        - **Estimated filter order N**: {N_est}
        {'- **Estimated β**: {:.2f}'.format(est_beta) if rec_win == 'Kaiser' else ''}
        - **Transition width factor**: {tw_factor:.2f} × fs/N
        """)

        st.latex(rf"N \approx \left\lceil {tw_factor:.2f} \times \frac{{f_s}}{{\Delta f}} \right\rceil = \left\lceil {tw_factor:.2f} \times \frac{{{fs_spec}}}{{{transition_width}}} \right\rceil = {N_est}")

    # ---- What-If Explorer ----
    with explore_tab2:
        st.markdown("#### Explore how parameters affect the filter")
        st.markdown("Adjust **N** and **β** below and see the real-time effect on the frequency response.")

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            wi_N = st.slider("Filter order N", 5, 201, 31, 2, key="wi_N")
            if wi_N % 2 == 0:
                wi_N += 1
        with col_w2:
            wi_window = st.selectbox("Window", WINDOW_NAMES, key="wi_win")
            wi_beta = 6.0
            if wi_window == "Kaiser":
                wi_beta = st.slider("β", 0.0, 14.0, 6.0, 0.5, key="wi_beta")

        # Compute
        if filter_type == "Lowpass":
            wi_hd = ideal_lowpass(wi_N, wc_rad)
        elif filter_type == "Highpass":
            wi_hd = ideal_highpass(wi_N, wc_rad)
        elif filter_type == "Bandpass":
            wi_hd = ideal_bandpass(wi_N, wc_rad, wc2_rad)
        else:
            wi_hd = ideal_bandstop(wi_N, wc_rad, wc2_rad)

        wi_w = get_window(wi_window, wi_N, wi_beta)
        wi_h = wi_hd * wi_w
        wi_omega, wi_H = freq_response(wi_h)
        wi_db = 20 * np.log10(np.maximum(np.abs(wi_H), 1e-12))

        fig_wi = go.Figure()
        fig_wi.add_trace(go.Scatter(
            x=wi_omega / np.pi, y=wi_db, mode="lines",
            line=dict(color=COLORS[wi_window], width=2),
            name=f"{wi_window}, N={wi_N}",
        ))
        if filter_type in ("Lowpass", "Highpass"):
            fig_wi.add_vline(x=wc, line_dash="dash", line_color="gray")
        else:
            fig_wi.add_vline(x=wc_rad / np.pi, line_dash="dash", line_color="gray")
            fig_wi.add_vline(x=wc2_rad / np.pi, line_dash="dash", line_color="gray")
        fig_wi.update_layout(**_plotly_layout(
            f"What-If: {wi_window} Window, N = {wi_N}", "Normalized Frequency (× π)", "Magnitude (dB)", 420))
        fig_wi.update_yaxes(range=[-120, 5])
        st.plotly_chart(fig_wi)

        # Show the impulse response too
        fig_wi_h = go.Figure()
        fig_wi_h.add_trace(go.Scatter(
            x=np.arange(wi_N), y=wi_h, mode="lines+markers",
            line=dict(color=COLORS[wi_window], width=2),
            marker=dict(size=4),
            name="h[n]",
        ))
        fig_wi_h.update_layout(**_plotly_layout(
            "Impulse Response h[n]", "Sample index n", "Amplitude", 350))
        st.plotly_chart(fig_wi_h)

    # ---- Quiz ----
    with explore_tab3:
        st.markdown("#### Self-Check Quiz")
        st.markdown("Test your understanding of the window method!")

        questions = [
            {
                "q": "1. Which window provides the **narrowest** main transition width?",
                "options": ["Rectangular", "Hann", "Hamming", "Blackman", "Kaiser (β=10)"],
                "answer": 0,
                "explanation": "The **Rectangular** window has the narrowest main lobe (TW factor = 0.91 fs/N), but this comes at the cost of high sidelobes (only 21 dB stopband attenuation)."
            },
            {
                "q": "2. Increasing the filter order N will:",
                "options": [
                    "Decrease the transition width",
                    "Increase the stopband attenuation",
                    "Both of the above",
                    "Neither — it only affects the passband"
                ],
                "answer": 0,
                "explanation": "Increasing N **decreases the transition width** (sharper transition). The stopband attenuation is determined by the **window choice**, not N."
            },
            {
                "q": "3. The Hamming window achieves approximately what stopband attenuation?",
                "options": ["21 dB", "44 dB", "55 dB", "75 dB"],
                "answer": 2,
                "explanation": "The Hamming window provides approximately **55 dB** of stopband attenuation, making it a good general-purpose choice."
            },
            {
                "q": "4. In the Kaiser window, increasing β will:",
                "options": [
                    "Narrow the main lobe and increase sidelobes",
                    "Widen the main lobe and reduce sidelobes",
                    "Have no effect on the main lobe",
                    "Only affect the passband ripple"
                ],
                "answer": 1,
                "explanation": "Increasing β in the Kaiser window **widens the main lobe** (wider transition) but **reduces the sidelobe level** (better stopband attenuation). It provides a tunable trade-off."
            },
            {
                "q": "5. Why must the filter order N typically be **odd** for a Type I FIR filter?",
                "options": [
                    "To ensure the filter is causal",
                    "To have an integer-valued group delay (symmetric center tap)",
                    "To reduce computational cost",
                    "It doesn't need to be odd"
                ],
                "answer": 1,
                "explanation": "An **odd** N gives a symmetric impulse response with an integer center sample, ensuring a Type I linear-phase FIR filter with **integer group delay**."
            },
        ]

        score = 0
        total = len(questions)
        answered = 0

        for i, qdata in enumerate(questions):
            st.markdown(f"**{qdata['q']}**")
            user_answer = st.radio(
                "Select your answer:",
                qdata["options"],
                key=f"quiz_{i}",
                index=None,
            )
            if user_answer is not None:
                answered += 1
                sel_idx = qdata["options"].index(user_answer)
                if sel_idx == qdata["answer"]:
                    st.success(f"✅ Correct! {qdata['explanation']}")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. {qdata['explanation']}")
            st.markdown("---")

        if answered == total:
            pct = score / total * 100
            if pct >= 80:
                st.balloons()
                st.success(f"🎉 Excellent! You scored **{score}/{total}** ({pct:.0f}%)")
            elif pct >= 60:
                st.warning(f"👍 Good effort! You scored **{score}/{total}** ({pct:.0f}%). Review the theory tab for topics you missed.")
            else:
                st.info(f"📚 You scored **{score}/{total}** ({pct:.0f}%). Spend some time with the Theory tab and try again!")
