"""
Multi-Food-Grain Quality Analysis: Local Web Demo
Project: Multi-Food-Grain Quality Analysis and Prediction Using Advanced Diffusion Models

Author: Senior ML Engineer & Python Developer
Usage:
    streamlit run app.py
"""

import os
import sys

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

from PIL import Image


def run_streamlit_app():
    from src.inference.predictor import RiceGrainPredictor, HUMAN_LABELS

    # 1. Page Configuration (Light Theme & Responsive)
    st.set_page_config(
        page_title="Multi-Food-Grain Quality Analysis",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Clean Custom CSS for Light Theme
    st.markdown("""
        <style>
        .main {
            background-color: #f8fafc;
            color: #1e293b;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: none;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
            color: white;
        }
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .disclaimer-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            color: #92400e;
            margin-top: 1.5rem;
        }
        .scope-box {
            background: #f1f5f9;
            border-left: 4px solid #64748b;
            padding: 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            color: #334155;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🌾 Multi-Food-Grain Quality Analysis")
    st.markdown("**Real-Time Rice Grain Defect Classification Demo** | MobileNetV3-Small Baseline (Session-Grouped)")
    st.divider()

    # Load Model (Cached Singleton)
    @st.cache_resource
    def load_predictor():
        ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "best_model.pth")
        return RiceGrainPredictor.get_instance(checkpoint_path=ckpt_path, device="cpu")

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        st.stop()

    # Layout: Two Columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Input Grain Image")
        
        # Quick Sample Selector
        sample_options = {
            "-- Upload my own image --": None,
            "Sample 1: Normal Grain (Sound)": "data/rice/test/0_NOR/Grainset_rice_2020-02-12-18-18-42_1_p600s.png",
            "Sample 2: Broken Grain": "data/rice/test/5_BN/Grainset_rice_2020-02-13-14-38-04_1_p600s.png",
            "Sample 3: Moldy Grain": "data/rice/test/3_MY/Grainset_rice_2020-02-13-16-11-20_4_p600s.png",
            "Sample 4: Pest-Attacked Grain": "data/rice/test/4_AP/Grainset_rice_2020-02-13-15-45-32_1_p600s.png",
            "Sample 5: Impurities (Chaff)": "data/rice/test/7_IM/Grainset_rice_2020-02-13-10-33-33_1_p600s.png"
        }
        
        selected_sample = st.selectbox("Choose a test sample or upload below:", list(sample_options.keys()))
        
        uploaded_file = st.file_uploader("Upload PNG or JPG image:", type=["png", "jpg", "jpeg"])
        
        active_image = None
        
        if uploaded_file is not None:
            try:
                active_image = Image.open(uploaded_file).convert("RGB")
                st.image(active_image, caption="Uploaded Grain Image", use_container_width=True)
            except Exception as e:
                st.error(f"Invalid image file: {e}")
        elif selected_sample and sample_options[selected_sample]:
            sample_path = os.path.join(PROJECT_ROOT, sample_options[selected_sample])
            if os.path.exists(sample_path):
                active_image = Image.open(sample_path).convert("RGB")
                st.image(active_image, caption=f"Selected: {selected_sample}", use_container_width=True)
            else:
                st.warning(f"Sample path not found: {sample_path}")

        predict_button = st.button("🔍 Predict Grain Quality")

    with col2:
        st.subheader("2. Quality Inspection Results")

        if predict_button:
            if active_image is None:
                st.warning("Please upload an image or choose a sample image first.")
            else:
                with st.spinner("Analyzing grain defect morphology..."):
                    result = predictor.predict(active_image)

                best_label = result["human_label"]
                best_code = result["predicted_code"]
                conf = result["confidence_percentage"]

                # Result Card
                st.markdown(f"""
                <div class="card">
                    <h4 style="margin: 0; color: #64748b; font-size: 0.9rem;">PRIMARY PREDICTION</h4>
                    <h2 style="margin: 0.2rem 0; color: #1e293b;">{best_label}</h2>
                    <p style="margin: 0; color: #475569; font-weight: 500;">
                        Dataset Category Code: <code>{best_code}</code> &nbsp;|&nbsp; Confidence: <strong>{conf:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(result["confidence"])

                # Top-3 Predictions Breakdown
                st.write("**Top-3 Class Probabilities:**")
                for rank, item in enumerate(result["top3_predictions"], start=1):
                    p_val = item["probability"]
                    pct = item["percentage"]
                    lbl = item["label"]
                    code = item["class_code"]
                    st.write(f"**{rank}. {lbl}** (`{code}`) — **{pct:.1f}%**")
                    st.progress(p_val)

        else:
            st.info("Upload or select an image on the left and click **Predict Grain Quality** to run inspection.")

        # Scope and Limitations Notice
        st.markdown("""
        <div class="scope-box">
            <strong>System Scope Notice:</strong><br>
            This demo classifies an isolated, cropped single-grain image into one of 8 GrainDet dataset categories.
            It does <em>not</em> currently perform:
            <ul style="margin: 0.3rem 0; padding-left: 1.2rem;">
                <li>Multi-grain tray object detection or automatic counting</li>
                <li>Sub-grain pixel defect segmentation (e.g. precise mold contouring)</li>
                <li>Laboratory moisture percentage measurement</li>
                <li>Official commercial grade assignment (Grade A, B, C, or Reject)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Research Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Research Disclaimer:</strong><br>
            Model predictions and confidence estimates are generated for academic research and experimental benchmarking.
            Confidence scores do not guarantee correctness and should not be used as the sole basis for commercial food-safety transactions.
        </div>
        """, unsafe_allow_html=True)


def main():
    if not _HAS_STREAMLIT:
        print("=" * 70)
        print("MULTI-FOOD-GRAIN QUALITY ANALYSIS - LOCAL WEB DEMO")
        print("=" * 70)
        print("[DEPENDENCY NOTICE] Streamlit is not installed in the active .venv.")
        print("\nTo launch this interactive web interface, please install Streamlit:")
        print("    pip install streamlit")
        print("\nAnd launch the demo using:")
        print("    streamlit run app.py")
        print("=" * 70)
    else:
        run_streamlit_app()


if __name__ == "__main__":
    main()
