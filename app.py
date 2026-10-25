"""
GrainSight AI: Multi-Food-Grain Quality Analysis Portal
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior ML Research Engineer

Features:
- MobileNetV3-Small real-time inference (~48ms on CPU)
- 8-Class Optical Rice Defect Classification
- Grad-CAM Explainable AI (XAI) defect attention heatmaps
- Commercial Quality Grading Engine (AGMARK & USDA Standards)
- Quantitative Morphology & Batch Lot Quality Simulator
- Multi-Food-Grain Species Taxonomy (Rice, Wheat, Sorghum)
"""

import os
import sys
import glob
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def run_streamlit_app():
    from src.inference.predictor import RiceGrainPredictor, HUMAN_LABELS
    from src.inference.grading import grade_single_kernel, compute_lot_grading, GRAIN_SPECIES_PROFILES

    # 1. Page Configuration (Light & Clean Theme)
    st.set_page_config(
        page_title="GrainSight AI | Multi-Food-Grain Quality",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Clean Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
            color: #1e293b;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .main-header {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.25rem 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            border: none;
            width: 100%;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
            color: white;
        }
        .card {
            background: white;
            padding: 1.25rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .disclaimer-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 0.85rem 1.2rem;
            border-radius: 6px;
            font-size: 0.825rem;
            color: #92400e;
            margin-top: 1rem;
        }
        .scope-box {
            background: #f1f5f9;
            border-left: 4px solid #64748b;
            padding: 0.85rem 1.2rem;
            border-radius: 6px;
            font-size: 0.825rem;
            color: #334155;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Header
    st.markdown("""
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; color: #0f172a; font-weight: 800;">🌾 GrainSight AI: Multi-Food-Grain Quality Analysis</h2>
                    <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.9rem;">
                        Real-Time Optical Defect Detection, Commercial AGMARK/USDA Grading &amp; Explainable AI (XAI)
                    </p>
                </div>
                <div>
                    <span class="badge" style="background: #e0e7ff; color: #4338ca;">v2.0 XAI-Enabled</span>
                    <span class="badge" style="background: #dcfce7; color: #15803d; margin-left: 0.3rem;">CPU Optimized</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Model Predictor Singleton
    @st.cache_resource
    def load_predictor():
        ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "best_model.pth")
        return RiceGrainPredictor.get_instance(checkpoint_path=ckpt_path, device="cpu")

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        st.stop()

    # 4. Sidebar: Multi-Species Taxonomy & Benchmark Metrics
    with st.sidebar:
        st.subheader("🌾 Multi-Grain Species Profile")
        selected_species = st.selectbox(
            "Select Grain Variety:",
            list(GRAIN_SPECIES_PROFILES.keys()),
            help="Choose cereal grain species to inspect grading specifications"
        )
        profile = GRAIN_SPECIES_PROFILES[selected_species]
        st.info(f"**Status**: {profile['status']}\n\n**Standard**: {profile['standards_body']}\n\n{profile['description']}")

        st.divider()

        st.subheader("📊 Engine Certified Metrics")
        st.markdown("""
        - **Architecture**: MobileNetV3-Small
        - **Input Resolution**: 224 × 224 (SquarePad)
        - **Inference Latency**: ~48 ms (Intel CPU)
        - **Held-Out Test Accuracy**: **89.74%**
        - **Macro F1-Score**: **0.7968**
        - **Held-Out Test Samples**: 4,953 Kernels
        - **Data Leakage**: 0.0% (Session-Grouped)
        """)

        st.divider()
        st.markdown("<small style='color: #94a3b8;'>GrainSight AI &bull; Proprietary Research Platform &bull; 2026</small>", unsafe_allow_html=True)

    # 5. Main Layout: Two Columns
    col1, col2 = st.columns([1, 1], gap="large")

    active_image = None
    selected_sample_key = None

    with col1:
        st.subheader("1. Input Grain Image & Settings")

        sample_options = {
            "-- Upload my own image --": None,
            "Sample 1: Normal Grain (Sound / Healthy)": "data/rice/test/0_NOR/Grainset_rice_2020-02-12-18-18-42_1_p600s.png",
            "Sample 2: Broken Grain (Mechanical Fracture)": "data/rice/test/5_BN/Grainset_rice_2020-02-13-14-38-04_1_p600s.png",
            "Sample 3: Moldy Grain (Toxic Fungal Decay)": "data/rice/test/3_MY/Grainset_rice_2020-02-13-16-11-20_4_p600s.png",
            "Sample 4: Sprouted Grain (Germinated Radicle)": "data/rice/test/2_SD/Grainset_rice_2020-02-12-18-18-42_0_p600s.png",
            "Sample 5: Pest-Attacked Grain (Insect Borehole)": "data/rice/test/4_AP/Grainset_rice_2020-02-13-15-45-32_1_p600s.png",
            "Sample 6: Fusarium Blight & Scab Damage": "data/rice/test/1_F&S/Grainset_rice_2020-02-14-14-55-35_0_p600s.png",
            "Sample 7: Impurities (Chaff & Foreign Matter)": "data/rice/test/7_IM/Grainset_rice_2020-02-13-10-33-33_1_p600s.png"
        }

        selected_sample_key = st.selectbox("Choose a sample preset or upload below:", list(sample_options.keys()))
        uploaded_file = st.file_uploader("Upload PNG or JPG grain photograph:", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            try:
                active_image = Image.open(uploaded_file).convert("RGB")
                st.image(active_image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"Invalid image file: {e}")
        elif selected_sample_key and sample_options[selected_sample_key]:
            sample_path = os.path.join(PROJECT_ROOT, sample_options[selected_sample_key])
            if os.path.exists(sample_path):
                active_image = Image.open(sample_path).convert("RGB")
                st.image(active_image, caption=f"Selected: {selected_sample_key}", use_container_width=True)
            else:
                st.warning(f"Sample path not found: {sample_path}")

        # Explainability & Diagnostics Toggle
        enable_gradcam = st.checkbox("🔍 Generate Grad-CAM Explainable AI (XAI) Defect Heatmap", value=True)

        predict_button = st.button("⚡ Run Quality & Defect Inspection")

    with col2:
        st.subheader("2. Quality & Commercial Grading Hub")

        if predict_button:
            if active_image is None:
                st.warning("Please upload an image or select a sample preset first.")
            else:
                with st.spinner("Executing neural defect classification and XAI analysis..."):
                    result = predictor.predict(active_image)
                    gradcam_res = None
                    if enable_gradcam:
                        gradcam_res = predictor.generate_gradcam(active_image)

                best_label = result["human_label"]
                best_code = result["predicted_code"]
                conf = result["confidence_percentage"]
                grading = result["grading"]

                # Commercial Grade Card (AGMARK / USDA)
                st.markdown(f"""
                <div class="card" style="border-left: 5px solid {grading['grade_color']};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge" style="background: {grading['grade_color']}20; color: {grading['grade_color']};">
                            {grading['grade_badge']} &bull; {grading['safety_status']}
                        </span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #64748b;">CODE: {best_code}</span>
                    </div>
                    <h3 style="margin: 0.4rem 0 0.2rem 0; color: #0f172a;">{best_label}</h3>
                    <p style="margin: 0 0 0.5rem 0; font-size: 0.875rem; color: #475569;">
                        <strong>Commercial Assessment:</strong> {grading['commercial_grade']}
                    </p>
                    <p style="margin: 0; font-size: 0.825rem; color: #64748b; line-height: 1.4;">
                        <strong>Actionable Advice:</strong> {grading['actionable_verdict']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Quantitative Metrics & Confidence Row
                w, h = active_image.size
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                q1, q2, q3 = st.columns(3)
                with q1:
                    st.metric("Model Certainty", f"{conf:.1f}%")
                with q2:
                    st.metric("Kernel Geometry", f"{w} × {h} px")
                with q3:
                    st.metric("Elongation (L/W)", f"{aspect_ratio:.2f}")

                st.progress(result["confidence"])

                # Grad-CAM XAI Visual Attention Display
                if gradcam_res is not None:
                    st.markdown("#### 🔬 Explainable AI: Grad-CAM Defect Attention Map")
                    st.caption("Highlights the exact morphological region of interest that triggered the classification.")
                    g_col1, g_col2 = st.columns(2)
                    with g_col1:
                        st.image(active_image.resize((224, 224)), caption="Original Optical Crop", use_container_width=True)
                    with g_col2:
                        st.image(gradcam_res["blended_image"], caption=f"Grad-CAM Heatmap ({best_code})", use_container_width=True)

                # Top-3 Class Probability Breakdown
                st.markdown("#### 📊 Top-3 Class Probabilities")
                for rank, item in enumerate(result["top3_predictions"], start=1):
                    p_val = item["probability"]
                    pct = item["percentage"]
                    lbl = item["label"]
                    code = item["class_code"]
                    st.write(f"**{rank}. {lbl}** (`{code}`) — **{pct:.1f}%**")
                    st.progress(p_val)

        else:
            st.info("Select or upload an optical image on the left and click **Run Quality & Defect Inspection**.")

    # 6. Interactive Section: Commercial Batch Lot Quality Simulator
    st.divider()
    st.subheader("3. Commercial Batch Lot Inspection Simulator (AGMARK & USDA Standards)")
    st.markdown("Simulate optical inspection across a commercial grain lot (20 random test kernels) to evaluate overall batch quality grade.")

    if st.button("🧪 Run Batch Lot Inspection Simulation"):
        with st.spinner("Scanning random commercial grain lot samples..."):
            test_dir = os.path.join(PROJECT_ROOT, "data", "rice", "test")
            all_test_images = glob.glob(os.path.join(test_dir, "*", "*.png"))
            
            import random
            random.seed(42)
            lot_sample_paths = random.sample(all_test_images, min(20, len(all_test_images)))
            
            batch_predictions = []
            for p in lot_sample_paths:
                res = predictor.predict(p)
                batch_predictions.append(res["predicted_code"])

            lot_report = compute_lot_grading(batch_predictions)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Sound Grain %", f"{lot_report['sound_percentage']}%")
        with b2:
            st.metric("Broken Grain %", f"{lot_report['broken_percentage']}%")
        with b3:
            st.metric("Biological Defect %", f"{lot_report['biological_defect_percentage']}%")
        with b4:
            st.metric("Foreign Matter %", f"{lot_report['impurity_percentage']}%")

        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {lot_report['grade_color']}; margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge" style="background: {lot_report['grade_color']}20; color: {lot_report['grade_color']};">
                    FINAL LOT GRADE: {lot_report['grade_badge']}
                </span>
                <span style="font-weight: 700; color: #475569;">Scanned: {lot_report['total_kernels']} Kernels</span>
            </div>
            <h3 style="margin: 0.4rem 0 0.2rem 0; color: #0f172a;">{lot_report['overall_lot_grade']}</h3>
            <p style="margin: 0; font-size: 0.85rem; color: #64748b;">
                <strong>Standard Compliance Verdict:</strong> {lot_report['compliance_verdict']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 7. Scope & Research Disclaimer
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("""
        <div class="scope-box">
            <strong>Industrial Scope Notice:</strong><br>
            Single-kernel classification runs on cropped optical streams. Production elevators integrate upstream multi-grain tray segmentation to extract individual grains before feeding this inference engine.
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Research &amp; Regulatory Disclaimer:</strong><br>
            Predictions, Grad-CAM attention maps, and commercial grade estimates are designed for academic research, automated pre-sorting, and experimental benchmarking under AGMARK / USDA frameworks.
        </div>
        """, unsafe_allow_html=True)


def main():
    if not _HAS_STREAMLIT:
        print("=" * 70)
        print("GRAINSIGHT AI: MULTI-FOOD-GRAIN QUALITY ANALYSIS PORTAL")
        print("=" * 70)
        print("Launch command:")
        print("    & '.\\.venv\\Scripts\\python.exe' -m streamlit run app.py --server.headless true --server.port 8501")
        print("=" * 70)
    else:
        run_streamlit_app()


if __name__ == "__main__":
    main()
