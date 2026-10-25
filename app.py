"""
GrainSight AI: Multi-Food-Grain Quality Analysis Portal
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior ML Research Engineer

A clean, production-grade inspection interface featuring:
- Optical Rice Defect Classification (8 classes, MobileNetV3-Small)
- Grad-CAM Explainable AI (XAI) attention heatmaps
- Commercial AGMARK & USDA Quality Grading
- One-Click Official PDF Inspection Report Export
"""

import os
import sys
import glob
import random
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
    from src.inference.predictor import RiceGrainPredictor
    from src.inference.grading import compute_lot_grading, GRAIN_SPECIES_PROFILES
    from src.inference.pdf_report import generate_inspection_pdf

    # 1. Page Configuration
    st.set_page_config(
        page_title="GrainSight AI | Grain Quality Portal",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Modern, Clean CSS Theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .main-header {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
            width: 100%;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #1d4ed8;
            color: white;
        }
        .stDownloadButton>button {
            background-color: #059669 !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 0.65rem 1.2rem !important;
            border: none !important;
            width: 100% !important;
            box-shadow: 0 2px 4px rgba(5,150,105,0.2) !important;
        }
        .stDownloadButton>button:hover {
            background-color: #047857 !important;
            color: white !important;
        }
        .card {
            background: #ffffff;
            padding: 1.25rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            margin-bottom: 1rem;
        }
        .grade-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. Top Header
    st.markdown("""
        <div class="main-header">
            <div>
                <h2 style="margin: 0; color: #0f172a; font-weight: 800; font-size: 1.5rem;">
                    🌾 GrainSight AI &bull; Grain Quality Inspection Portal
                </h2>
                <p style="margin: 0.2rem 0 0 0; color: #64748b; font-size: 0.875rem;">
                    Real-Time Defect Detection, Commercial AGMARK/USDA Grading &amp; Inspection Reports
                </p>
            </div>
            <div>
                <span class="grade-badge" style="background: #e0e7ff; color: #3730a3;">v2.0 XAI Engine</span>
                <span class="grade-badge" style="background: #dcfce7; color: #166534; margin-left: 0.4rem;">CPU Optimized</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 4. Model Predictor Singleton
    @st.cache_resource
    def load_predictor():
        ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "baseline", "best_model.pth")
        return RiceGrainPredictor.get_instance(checkpoint_path=ckpt_path, device="cpu")

    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        st.stop()

    # 5. Sidebar: Settings & Specifications
    with st.sidebar:
        st.markdown("### 🌾 Inspection Settings")
        selected_species = st.selectbox(
            "Grain Variety:",
            list(GRAIN_SPECIES_PROFILES.keys()),
            help="Select grain species profile for commercial grading standards"
        )
        profile = GRAIN_SPECIES_PROFILES[selected_species]
        st.caption(f"**Standard:** {profile['standards_body']} &bull; **Status:** {profile['status']}")

        st.divider()

        st.markdown("### 📊 Engine Specifications")
        st.markdown("""
        - **Model**: MobileNetV3-Small (SquarePad)
        - **Inference Speed**: ~48 ms / grain (CPU)
        - **Model Accuracy**: **89.74%**
        - **Macro F1-Score**: **0.7968**
        - **Defect Categories**: 8 Classes
        """)

        st.divider()
        st.markdown("<small style='color: #94a3b8;'>GrainSight AI &bull; Inspection System</small>", unsafe_allow_html=True)

    # 6. Main Columns Layout
    col1, col2 = st.columns([1, 1], gap="large")

    active_image = None

    with col1:
        st.markdown("### 1. Select or Upload Grain Image")

        sample_options = {
            "-- Upload custom image --": None,
            "Sample 1: Normal Grain (Sound / Healthy)": "data/rice/test/0_NOR/Grainset_rice_2020-02-12-18-18-42_1_p600s.png",
            "Sample 2: Broken Grain (Mechanical Fracture)": "data/rice/test/5_BN/Grainset_rice_2020-02-13-14-38-04_1_p600s.png",
            "Sample 3: Moldy Grain (Toxic Fungal Decay)": "data/rice/test/3_MY/Grainset_rice_2020-02-13-16-11-20_4_p600s.png",
            "Sample 4: Sprouted Grain (Germinated Radicle)": "data/rice/test/2_SD/Grainset_rice_2020-02-12-18-18-42_0_p600s.png",
            "Sample 5: Pest-Attacked Grain (Insect Borehole)": "data/rice/test/4_AP/Grainset_rice_2020-02-13-15-45-32_1_p600s.png",
            "Sample 6: Fusarium Blight & Scab Damage": "data/rice/test/1_F&S/Grainset_rice_2020-02-14-14-55-35_0_p600s.png",
            "Sample 7: Impurities (Chaff & Foreign Matter)": "data/rice/test/7_IM/Grainset_rice_2020-02-13-10-33-33_1_p600s.png"
        }

        selected_sample_key = st.selectbox("Preset defect samples:", list(sample_options.keys()))
        uploaded_file = st.file_uploader("Or upload image (PNG / JPG):", type=["png", "jpg", "jpeg"])

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
                st.image(active_image, caption=selected_sample_key, use_container_width=True)
            else:
                st.warning(f"Sample not found: {sample_path}")

        enable_gradcam = st.checkbox("Generate Grad-CAM Defect Attention Map (XAI)", value=True)

        analyze_clicked = st.button("🔍 Inspect Grain Quality")

        # Maintain state so download button does not clear results
        if analyze_clicked:
            if active_image is None:
                st.warning("Please upload an image or choose a preset sample first.")
            else:
                with st.spinner("Analyzing grain morphological defects..."):
                    res = predictor.predict(active_image)
                    cam_res = predictor.generate_gradcam(active_image) if enable_gradcam else None
                    st.session_state["last_result"] = res
                    st.session_state["last_cam"] = cam_res
                    st.session_state["last_img"] = active_image

    with col2:
        st.markdown("### 2. Quality & Commercial Grading Results")

        if "last_result" in st.session_state and st.session_state["last_result"] is not None:
            result = st.session_state["last_result"]
            gradcam_res = st.session_state.get("last_cam")
            analyzed_img = st.session_state.get("last_img")

            best_label = result["human_label"]
            best_code = result["predicted_code"]
            conf = result["confidence_percentage"]
            grading = result["grading"]

            # Commercial Grade Summary Card
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid {grading['grade_color']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="grade-badge" style="background: {grading['grade_color']}20; color: {grading['grade_color']};">
                        {grading['grade_badge']} &bull; {grading['safety_status']}
                    </span>
                    <span style="font-size: 0.8rem; font-weight: 700; color: #64748b;">CODE: {best_code}</span>
                </div>
                <h3 style="margin: 0.5rem 0 0.2rem 0; color: #0f172a;">{best_label}</h3>
                <p style="margin: 0 0 0.4rem 0; font-size: 0.875rem; color: #334155;">
                    <strong>Commercial Standard:</strong> {grading['commercial_grade']}
                </p>
                <p style="margin: 0; font-size: 0.825rem; color: #64748b; line-height: 1.4;">
                    <strong>Recommendation:</strong> {grading['actionable_verdict']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Quantitative Metrics
            w, h = analyzed_img.size
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Model Confidence", f"{conf:.1f}%")
            with q2:
                st.metric("Kernel Geometry", f"{w} × {h} px")
            with q3:
                st.metric("Aspect Ratio (L/W)", f"{aspect_ratio:.2f}")

            # Grad-CAM Defect Attention Map Display
            if gradcam_res is not None:
                st.markdown("#### 🔬 Explainable AI: Defect Attention Heatmap")
                st.caption("Visual confirmation of morphological defect detected by the model.")
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.image(analyzed_img.resize((224, 224)), caption="Optical Grain Crop", use_container_width=True)
                with g_col2:
                    st.image(gradcam_res["blended_image"], caption=f"Defect Hotspot ({best_code})", use_container_width=True)

            # Top 3 Predicted Defect Classes
            st.markdown("#### 📊 Top Defect Probabilities")
            for rank, item in enumerate(result["top3_predictions"], start=1):
                st.write(f"**{rank}. {item['label']}** (`{item['class_code']}`) — **{item['percentage']:.1f}%**")
                st.progress(item["probability"])

            # One-Click Official PDF Report Download
            st.divider()
            st.markdown("#### 📄 Official Inspection Certificate")
            try:
                cam_pil = gradcam_res["blended_image"] if gradcam_res is not None else None
                pdf_bytes = generate_inspection_pdf(
                    original_pil=analyzed_img,
                    gradcam_pil=cam_pil,
                    prediction_result=result,
                    grain_species=selected_species
                )
                report_filename = f"GrainSight_Report_{best_code}_{random.randint(1000, 9999)}.pdf"
                st.download_button(
                    label="📥 Download Official Inspection Report (PDF)",
                    data=pdf_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")

        else:
            st.info("👈 Select a sample grain or upload an image and click **Inspect Grain Quality** to see detailed quality grading and export the PDF report.")

    # 7. Optional Collapsible Commercial Batch Lot Simulator
    st.divider()
    with st.expander("🧪 Commercial Batch Lot Quality Simulator (AGMARK & USDA Standards)", expanded=False):
        st.markdown("Simulate optical inspection across a commercial batch (20 random test kernels) to evaluate aggregate grade.")
        if st.button("Run Batch Lot Simulation"):
            with st.spinner("Scanning commercial lot kernels..."):
                test_dir = os.path.join(PROJECT_ROOT, "data", "rice", "test")
                all_test_images = glob.glob(os.path.join(test_dir, "*", "*.png"))
                
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
                st.metric("Defect Grain %", f"{lot_report['biological_defect_percentage']}%")
            with b4:
                st.metric("Foreign Matter %", f"{lot_report['impurity_percentage']}%")

            st.markdown(f"""
            <div class="card" style="border-left: 5px solid {lot_report['grade_color']}; margin-top: 0.5rem;">
                <span class="grade-badge" style="background: {lot_report['grade_color']}20; color: {lot_report['grade_color']};">
                    BATCH GRADE: {lot_report['grade_badge']}
                </span>
                <h4 style="margin: 0.4rem 0 0.2rem 0; color: #0f172a;">{lot_report['overall_lot_grade']}</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #64748b;">
                    <strong>Compliance Verdict:</strong> {lot_report['compliance_verdict']}
                </p>
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
