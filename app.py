import streamlit as st
import cv2
import numpy as np
import os
import sys

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.report import generate_full_report
from src.histogram import plot_histogram

# Page Configuration
st.set_page_config(
    page_title="Smart Image Quality Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (CSS)
st.markdown(
    """
    <style>
    /* Dark Theme Customizations */
    .stApp {
        background-color: #0f111a;
        color: #e0e6ed;
    }
    .header-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3b3b54;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: #181825;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #89b4fa;
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #a6adc8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #cdd6f4;
    }
    .metric-status {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-good { color: #a6e3a1; background: rgba(166,227,161,0.1); padding: 4px 10px; border-radius: 12px; display: inline-block; }
    .badge-avg { color: #f9e2af; background: rgba(249,226,175,0.1); padding: 4px 10px; border-radius: 12px; display: inline-block; }
    .badge-bad { color: #f38ba8; background: rgba(243,139,168,0.1); padding: 4px 10px; border-radius: 12px; display: inline-block; }

    /* Score gauge container */
    .score-box {
        text-align: center;
        background: radial-gradient(circle at center, #1e1e2e 0%, #11111b 100%);
        border: 2px solid #45475a;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .score-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    """
    <div class="header-card">
        <h1 style="margin:0; font-size:2.2rem; color:#89b4fa;">🔍 Smart Image Quality Analyzer</h1>
        <p style="margin-top:8px; margin-bottom:0; color:#bac2de; font-size:1.05rem;">
            An Academic Computer Vision tool for objective, rule-based image quality evaluation, contrast measurement, blur detection, noise estimation, and histogram analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Options
st.sidebar.header("📥 Image Input & Options")

# Sample Image Presets
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "assets", "sample_images")
sample_files = {
    "None (Upload Custom)": None,
    "Sample 1: Good Quality Image": os.path.join(SAMPLE_DIR, "sample_good.jpg"),
    "Sample 2: Dark & Blurry Image": os.path.join(SAMPLE_DIR, "sample_dark_blurry.jpg"),
    "Sample 3: High Noise Image": os.path.join(SAMPLE_DIR, "sample_noisy.jpg"),
}

selected_sample = st.sidebar.selectbox("Select Sample Preset", list(sample_files.keys()))

uploaded_file = st.sidebar.file_uploader(
    "Or Upload an Image (JPG, JPEG, PNG, BMP)",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload an image file to analyze its visual parameters.",
)

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Processing Options")
show_equalized = st.sidebar.checkbox("Show Histogram Equalization Comparison", value=True)
show_raw_deductions = st.sidebar.checkbox("Show Transparent Score Breakdown", value=True)

# Determine input source
source = None
if uploaded_file is not None:
    source = uploaded_file
elif sample_files[selected_sample] is not None:
    source = sample_files[selected_sample]

if source is None:
    st.info("👈 Please select a sample image preset or upload an image from the sidebar to begin analysis.")
    
    # Showcase visual placeholder card
    st.markdown(
        """
        <div style="background:#181825; border:1px dashed #45475a; border-radius:12px; padding:40px; text-align:center; margin-top:20px;">
            <h3 style="color:#cdd6f4;">No Image Loaded</h3>
            <p style="color:#a6adc8;">Upload a JPG, JPEG, PNG, or BMP file using the sidebar panel on the left.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Academic CV Concepts Section
    with st.expander("📚 Learn About the Computer Vision Concepts Used", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                #### 1. Brightness & Mean Intensity
                Brightness is computed as the first moment (mean) of the 8-bit grayscale intensity distribution:
                $$\\mu = \\frac{1}{N} \\sum_{i=1}^{N} I(i)$$
                Values below 40 indicate severe underexposure, while values above 220 indicate highlight clipping.

                #### 2. Contrast & Standard Deviation
                Contrast evaluates the dispersion of pixel intensities across the dynamic range using standard deviation:
                $$\\sigma = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (I(i) - \\mu)^2}$$
                Higher standard deviation signifies richer contrast between shadow and highlight regions.
                """
            )
        with col_b:
            st.markdown(
                """
                #### 3. Sharpness via Laplacian Variance
                Sharp edges produce high spatial gradients. The Laplacian operator $\\Delta I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}$ highlights high-frequency details. Blur reduces edge intensity, resulting in low Laplacian variance:
                $$\\text{Sharpness Score} = \\text{Var}(\\Delta I)$$

                #### 4. High-Frequency Noise Residual
                Noise level is estimated by subtracting a Gaussian smoothed image $G_\\sigma(I)$ from the original grayscale image $I$:
                $$I_{\\text{residual}} = I - G_\\sigma(I)$$
                The standard deviation $\\sigma_{\\text{residual}}$ quantifies high-frequency zero-mean sensor noise.
                """
            )

else:
    # Run Computer Vision Analysis Pipeline with Error Handling
    try:
        report = generate_full_report(source)
    except ValueError as e:
        st.error(f"⚠️ Image Processing Error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred while processing the image: {str(e)}")
        st.stop()

    # Extract results
    img_bgr = report["img_bgr"]
    gray_img = report["gray_img"]
    equalized_img = report["equalized_img"]
    meta = report["metadata"]
    b_res = report["brightness"]
    c_res = report["contrast"]
    s_res = report["sharpness"]
    n_res = report["noise"]
    q_res = report["quality"]

    # Top Row: Quality Score Gauge & Metadata
    col_score, col_meta = st.columns([1, 1.2])

    with col_score:
        score_val = q_res["overall_score"]
        category = q_res["category"]

        if category == "Good Quality":
            score_color = "#a6e3a1"
            badge_class = "badge-good"
        elif category == "Average Quality":
            score_color = "#f9e2af"
            badge_class = "badge-avg"
        else:
            score_color = "#f38ba8"
            badge_class = "badge-bad"

        st.markdown(
            f"""
            <div class="score-box">
                <div class="metric-title">Overall Quality Rating</div>
                <div class="score-number" style="color: {score_color};">{score_val}<span style="font-size:1.8rem;">/100</span></div>
                <div style="margin-top:10px;">
                    <span class="{badge_class}">{category}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_meta:
        st.markdown("### 📐 Image Metadata")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Dimensions", f"{meta['width']} × {meta['height']} px")
            st.metric("Channels", f"{meta['channels']} ({'Color BGR' if meta['channels']==3 else 'Grayscale'})")
        with m_col2:
            st.metric("Total Pixels", f"{meta['total_pixels']:,}")
            st.metric("Aspect Ratio", f"{meta['aspect_ratio']}:1")

    # Metrics Breakdown Row (4 Cards)
    st.markdown("---")
    st.markdown("### 📊 Core Metric Parameters")

    m_col_1, m_col_2, m_col_3, m_col_4 = st.columns(4)

    def get_status_badge(status_str):
        if any(w in status_str for w in ["Optimal", "Good", "Sharp", "Low Noise"]):
            return f'<span class="badge-good">{status_str}</span>'
        elif any(w in status_str for w in ["Slightly", "Moderate"]):
            return f'<span class="badge-avg">{status_str}</span>'
        else:
            return f'<span class="badge-bad">{status_str}</span>'

    with m_col_1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💡 Brightness</div>
                <div class="metric-value">{b_res['brightness_score']}</div>
                <div class="metric-status">{get_status_badge(b_res['status'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col_2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">☯️ Contrast</div>
                <div class="metric-value">{c_res['contrast_score']}</div>
                <div class="metric-status">{get_status_badge(c_res['status'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col_3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🎯 Sharpness</div>
                <div class="metric-value">{s_res['sharpness_score']}</div>
                <div class="metric-status">{get_status_badge(s_res['status'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col_4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📻 Noise Residual</div>
                <div class="metric-value">{n_res['noise_score']}</div>
                <div class="metric-status">{get_status_badge(n_res['status'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Transparent Score Breakdown Expander
    if show_raw_deductions:
        with st.expander("🔍 Score Deduction Formula & Breakdown", expanded=False):
            st.markdown(
                """
                **Rule-Based Scoring Formula**:  
                Points start at **100.0**. Deductions are penalised transparently based on metric deviations:
                """
            )
            d_cols = st.columns(4)
            d_cols[0].metric("Brightness Penalty", f"-{q_res['deductions']['brightness']} pts")
            d_cols[1].metric("Contrast Penalty", f"-{q_res['deductions']['contrast']} pts")
            d_cols[2].metric("Sharpness Penalty", f"-{q_res['deductions']['sharpness']} pts")
            d_cols[3].metric("Noise Penalty", f"-{q_res['deductions']['noise']} pts")

    # Visual Comparison & Image Display
    st.markdown("---")
    st.markdown("### 🖼️ Image & Histogram Analysis")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if show_equalized:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(img_rgb, caption="Original Input Image", use_container_width=True)
        with img_col2:
            st.image(equalized_img, caption="Histogram Equalized (Enhanced Contrast)", use_container_width=True)
    else:
        st.image(img_rgb, caption="Original Input Image", use_container_width=True)

    # Histogram Chart
    st.markdown("#### 📉 Grayscale Intensity Histogram")
    fig_hist = plot_histogram(gray_img, eq_img=equalized_img if show_equalized else None)
    st.pyplot(fig_hist)

    # Recommendations Section
    st.markdown("---")
    st.markdown("### 💡 Practical Recommendations")

    if q_res["category"] == "Good Quality":
        st.success("🎉 **High Quality Image**: No significant defects detected.")
    else:
        st.warning(f"⚠️ **Attention Required**: Image classified as **{q_res['category']}**.")

    for rec in report["recommendations"]:
        st.info(rec)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#6c7086; font-size:0.85rem;">
            Academic Computer Vision Project — Smart Image Quality Analyzer | Built with Python, OpenCV, NumPy & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
