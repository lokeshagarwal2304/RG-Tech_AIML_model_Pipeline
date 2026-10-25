"""
GrainSight AI: Commercial Quality Grading Engine & Quantitative Analysis
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior ML Research Engineer

Implements official industrial grain grading standards (USDA / AGMARK / ISO 7301):
- Single-Kernel Commercial Grading & Defect Severity
- Batch Lot Quality Grading (Broken %, Defect %, Impurity %, Lot Assignment)
- Multi-Food-Grain Species Profiles (Rice, Wheat, Sorghum)
"""

# Multi-Food-Grain Taxonomy Profiles
GRAIN_SPECIES_PROFILES = {
    "Rice (Oryza sativa)": {
        "status": "Active Trained AI Engine",
        "description": "Dual-camera optical inspection with 8-class defect taxonomy.",
        "standards_body": "AGMARK & USDA Commercial Rice Standards",
        "classes": {
            "0_NOR": "Sound / Healthy Kernel",
            "1_F&S": "Fusarium Head Blight & Scab",
            "2_SD":  "Sprouted / Germinated Kernel",
            "3_MY":  "Moldy / Fungal Kernel",
            "4_AP":  "Pest / Insect Borehole",
            "5_BN":  "Broken Kernel (Fractured)",
            "6_UN":  "Unripe / Chalky Immature",
            "7_IM":  "Chaff & Foreign Organic Matter"
        }
    },
    "Wheat (Triticum aestivum)": {
        "status": "AV4GA Benchmark Profile",
        "description": "Cereal grain defect profile based on AV4GA dual-view specifications.",
        "standards_body": "USDA Grain Inspection, Packers and Stockyards Administration (GIPSA)",
        "classes": {
            "0_NOR": "Sound Wheat Grain",
            "1_FS":  "Fusarium & Scab Damaged",
            "2_SD":  "Sprouted Wheat Kernel",
            "3_BP":  "Black Germ / Black Point",
            "4_MY":  "Moldy Wheat Kernel",
            "5_BN":  "Broken Wheat Kernel",
            "6_AP":  "Insect-Bitten Wheat Grain",
            "7_IM":  "Non-Grain Impurities"
        }
    },
    "Sorghum (Sorghum bicolor)": {
        "status": "GrainSpace Benchmark Profile",
        "description": "Coarse cereal quality inspection standard for industrial milling.",
        "standards_body": "Codex Alimentarius Standard for Sorghum Grains (CXS 172-1989)",
        "classes": {
            "0_NOR": "Sound Sorghum Kernel",
            "1_FS":  "Fungal Head Blight",
            "2_SD":  "Sprouted Sorghum",
            "3_HD":  "Hard Endosperm Defect",
            "4_MY":  "Mold-Contaminated Sorghum",
            "5_BN":  "Broken Sorghum Kernel",
            "6_AP":  "Insect Damaged Sorghum",
            "7_IM":  "Chaff & Stones"
        }
    }
}


def grade_single_kernel(predicted_code, confidence):
    """Computes single-kernel commercial grade and actionable advice."""
    if predicted_code == "0_NOR":
        return {
            "commercial_grade": "Grade A (Sound / Premium)",
            "grade_badge": "GRADE A",
            "grade_color": "#10b981", # Emerald
            "defect_category": "Sound Kernel (Zero Defect)",
            "actionable_verdict": "Kernel satisfies Grade A export specifications. Approved for direct food processing.",
            "safety_status": "SAFE"
        }
    elif predicted_code == "5_BN":
        return {
            "commercial_grade": "Grade B (Mechanical Fracture)",
            "grade_badge": "GRADE B",
            "grade_color": "#3b82f6", # Blue
            "defect_category": "Mechanical Fracture",
            "actionable_verdict": "Sound kernel fractured during threshing/milling. Divert to broken-rice processing.",
            "safety_status": "COMMERCIAL DIVERTER"
        }
    elif predicted_code == "6_UN":
        return {
            "commercial_grade": "Grade B (Chalky / Immature)",
            "grade_badge": "GRADE B",
            "grade_color": "#6366f1", # Indigo
            "defect_category": "Physiological Immaturity",
            "actionable_verdict": "Kernel exhibits premature harvesting chalkiness. Acceptable for domestic cooking grades.",
            "safety_status": "COMMERCIAL PASS"
        }
    elif predicted_code == "7_IM":
        return {
            "commercial_grade": "Reject (Foreign Contaminant)",
            "grade_badge": "REJECT",
            "grade_color": "#f59e0b", # Amber
            "defect_category": "Non-Grain Foreign Matter",
            "actionable_verdict": "Foreign organic material detected. Requires pneumatic aspiration and destoning.",
            "safety_status": "CONTAMINANT"
        }
    elif predicted_code in ["1_F&S", "3_MY"]:
        return {
            "commercial_grade": "Reject (Toxic Fungal Blight)",
            "grade_badge": "REJECT",
            "grade_color": "#ef4444", # Red
            "defect_category": "Biological Fungal Pathogen",
            "actionable_verdict": "Hazardous fungal mycelium detected. High risk of mycotoxins; mandatory optical rejection.",
            "safety_status": "HAZARDOUS"
        }
    else:  # 2_SD, 4_AP
        return {
            "commercial_grade": "Reject (Degraded Kernel)",
            "grade_badge": "REJECT",
            "grade_color": "#dc2626", # Dark Red
            "defect_category": "Biological Germination / Insect Borehole",
            "actionable_verdict": "Kernel exhibits germination or insect pitting. Unfit for Grade A/B commercial lots.",
            "safety_status": "DEFECTIVE"
        }


def compute_lot_grading(class_codes_list):
    """Aggregates multiple kernel inspections into a certified commercial lot grade.
    
    Standards (AGMARK / USDA Rice Standards):
    - Grade A: Sound >= 90%, Broken <= 5%, Defect <= 3%, Foreign Matter == 0%
    - Grade B: Sound >= 80%, Broken <= 10%, Defect <= 6%, Foreign Matter <= 1%
    - Grade C: Sound >= 65%, Broken <= 18%, Defect <= 12%, Foreign Matter <= 2%
    - Reject : Defect > 12% OR Foreign Matter > 2% OR Mold > 2%
    """
    total = len(class_codes_list)
    if total == 0:
        return None

    counts = {
        "0_NOR": 0, "1_F&S": 0, "2_SD": 0, "3_MY": 0,
        "4_AP": 0, "5_BN": 0, "6_UN": 0, "7_IM": 0
    }
    for c in class_codes_list:
        if c in counts:
            counts[c] += 1

    sound_pct = (counts["0_NOR"] / total) * 100
    broken_pct = (counts["5_BN"] / total) * 100
    impurity_pct = (counts["7_IM"] / total) * 100
    bio_defect_count = counts["1_F&S"] + counts["2_SD"] + counts["3_MY"] + counts["4_AP"] + counts["6_UN"]
    bio_defect_pct = (bio_defect_count / total) * 100
    mold_pct = (counts["3_MY"] / total) * 100

    # Grading Rule Hierarchy
    if mold_pct > 2.0 or bio_defect_pct > 12.0 or impurity_pct > 2.0:
        lot_grade = "Reject (Non-Compliant / Contaminated)"
        badge = "REJECT"
        color = "#ef4444"
        verdict = "Lot rejected due to excessive biological decay or foreign contaminants. Mandatory reprocessing."
    elif sound_pct >= 90.0 and broken_pct <= 5.0 and impurity_pct == 0.0 and bio_defect_pct <= 3.0:
        lot_grade = "Grade A (Premium Export Quality)"
        badge = "GRADE A"
        color = "#10b981"
        verdict = "Exceptional batch quality. Satisfies international Grade A export benchmarks."
    elif sound_pct >= 80.0 and broken_pct <= 10.0 and impurity_pct <= 1.0 and bio_defect_pct <= 6.0:
        lot_grade = "Grade B (Standard Commercial Grade)"
        badge = "GRADE B"
        color = "#3b82f6"
        verdict = "Standard commercial grade. Approved for domestic wholesale and retail packaging."
    elif sound_pct >= 65.0 and broken_pct <= 18.0:
        lot_grade = "Grade C (Industrial / Animal Feed Grade)"
        badge = "GRADE C"
        color = "#f59e0b"
        verdict = "Substantial fraction of broken/chalky kernels. Divert to industrial ethanol or feed milling."
    else:
        lot_grade = "Reject (Substandard)"
        badge = "REJECT"
        color = "#dc2626"
        verdict = "Excessive grain damage exceeds allowable commercial thresholds."

    return {
        "total_kernels": total,
        "sound_percentage": round(sound_pct, 1),
        "broken_percentage": round(broken_pct, 1),
        "impurity_percentage": round(impurity_pct, 1),
        "biological_defect_percentage": round(bio_defect_pct, 1),
        "mold_percentage": round(mold_pct, 1),
        "overall_lot_grade": lot_grade,
        "grade_badge": badge,
        "grade_color": color,
        "compliance_verdict": verdict,
        "counts": counts
    }
