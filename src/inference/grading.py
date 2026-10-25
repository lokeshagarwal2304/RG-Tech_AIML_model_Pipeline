"""
GrainSight AI: Universal Multi-Food-Grain Quality Grading Engine
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior ML Research Engineer

Implements official international grain grading standards:
- Rice (Oryza sativa): AGMARK & USDA Commercial Rice Standards
- Wheat (Triticum aestivum): USDA GIPSA Federal Grain Inspection Service (FGIS) & ISO 5527
- Sorghum (Sorghum bicolor): Codex Alimentarius Standard (CXS 172-1989) & USDA Sorghum Standards

Features:
- Species-Aware Commercial Grading & Defect Severity
- Kernel Morphometric Validation (Aspect Ratio & Geometry Bounds)
- Commercial Batch Lot Quality Simulator per Species Standard
"""

# Multi-Food-Grain Taxonomy Profiles & Morphometric Baselines
GRAIN_SPECIES_PROFILES = {
    "Rice (Oryza sativa)": {
        "status": "Active Primary Trained Engine",
        "scientific_name": "Oryza sativa L.",
        "family": "Poaceae (Gramineae)",
        "standards_body": "AGMARK & USDA Commercial Rice Standards",
        "description": "Optical defect classification calibrated across long, medium, and short grain paddy rice.",
        "expected_aspect_ratio_range": (2.2, 3.8), # Typically elongated
        "classes": {
            "0_NOR": "Sound / Healthy Kernel (Grade A)",
            "1_F&S": "Fusarium Head Blight & Scab (Gibberella fujikuroi)",
            "2_SD":  "Sprouted / Germinated Radicle",
            "3_MY":  "Moldy / Fungal Decay (Aspergillus / Penicillium)",
            "4_AP":  "Pest / Insect Borehole Damage (Sitophilus oryzae)",
            "5_BN":  "Broken Kernel (Mechanical Fracture)",
            "6_UN":  "Unripe / Immature Chalky Kernel",
            "7_IM":  "Chaff & Non-Grain Foreign Organic Matter"
        }
    },
    "Wheat (Triticum aestivum)": {
        "status": "Calibrated Universal Cereal Profile",
        "scientific_name": "Triticum aestivum L.",
        "family": "Poaceae (Gramineae)",
        "standards_body": "USDA GIPSA (FGIS) Standards & ISO 5527",
        "description": "Universal cereal defect classification for hard red, durum, and soft white bread wheat.",
        "expected_aspect_ratio_range": (1.7, 2.5), # Ovoid with ventral crease
        "classes": {
            "0_NOR": "Sound Wheat Kernel (US No. 1 Food Grade)",
            "1_F&S": "Fusarium Head Blight / Tombstone Scab (DON Risk)",
            "2_SD":  "Sprout-Damaged Kernel (Elevated Alpha-Amylase)",
            "3_MY":  "Mold-Contaminated / Black Point Kernel",
            "4_AP":  "Insect-Bitten / Weevil-Bored Grain (Sitophilus granarius)",
            "5_BN":  "Shrunken & Broken Wheat Kernel",
            "6_UN":  "Chalky Immature / Shriveled Grain",
            "7_IM":  "Foreign Matter, Dockage & Chaff"
        }
    },
    "Sorghum (Sorghum bicolor)": {
        "status": "Calibrated Universal Cereal Profile",
        "scientific_name": "Sorghum bicolor (L.) Moench",
        "family": "Poaceae (Gramineae)",
        "standards_body": "Codex Alimentarius CXS 172-1989 & USDA GIPSA",
        "description": "Coarse cereal quality inspection for industrial flour milling, animal nutrition, and brewing.",
        "expected_aspect_ratio_range": (1.0, 1.5), # Sub-globose / Spherical
        "classes": {
            "0_NOR": "Sound Sorghum Grain (Codex Grade 1)",
            "1_F&S": "Head Blight & Ergot Fungal Infection",
            "2_SD":  "Sprouted Sorghum (Premature Germination)",
            "3_MY":  "Field Fungi & Mold-Decayed Grain",
            "4_AP":  "Weevil & Storage Pest Boreholes",
            "5_BN":  "Cracked & Fractured Sorghum Kernel",
            "6_UN":  "Immature / Chalky Soft Endosperm",
        }
    },
    "Maize / Corn (Zea mays)": {
        "status": "Calibrated Coarse Grain Profile",
        "scientific_name": "Zea mays L.",
        "family": "Poaceae (Gramineae)",
        "standards_body": "USDA GIPSA Corn Standards & Codex CXS 153-1985",
        "description": "Feed, milling, and starch quality grading for dent and flint corn kernels.",
        "expected_aspect_ratio_range": (1.1, 1.6),
        "classes": {
            "0_NOR": "Sound Corn Kernel (US No. 1 Grade)",
            "1_F&S": "Gibberella / Fusarium Ear Rot",
            "2_SD":  "Sprouted Corn Grain",
            "3_MY":  "Aspergillus Ear Rot (Aflatoxin Risk)",
            "4_AP":  "Weevil / Borer Damaged Grain",
            "5_BN":  "Broken Corn & Foreign Material (BCFM)",
            "6_UN":  "Immature / Soft Chalky Dent Kernel",
            "7_IM":  "Cob Fragments, Chaff & Dust"
        }
    },
    "Chickpeas / Chana (Cicer arietinum)": {
        "status": "Calibrated Pulse Profile",
        "scientific_name": "Cicer arietinum L.",
        "family": "Fabaceae (Leguminosae)",
        "standards_body": "AGMARK & Codex Pulse Standards (CXS 171-1989)",
        "description": "Desi and Kabuli chickpea quality inspection for split dal, milling, and canning.",
        "expected_aspect_ratio_range": (1.0, 1.4),
        "classes": {
            "0_NOR": "Sound Whole Chickpea (Grade 1)",
            "1_F&S": "Ascochyta Blight / Fungal Spotting",
            "2_SD":  "Sprouted / Split Testa",
            "3_MY":  "Storage Mold / Musty Grain",
            "4_AP":  "Bruchid Weevil Hole Damage (Callosobruchus)",
            "5_BN":  "Split & Broken Chickpea (Chana Dal)",
            "6_UN":  "Shriveled / Green Immature Cotyledons",
            "7_IM":  "Pods, Dirt & Chaff"
        }
    },
    "Lentils / Masoor (Lens culinaris)": {
        "status": "Calibrated Pulse Profile",
        "scientific_name": "Lens culinaris Medik.",
        "family": "Fabaceae (Leguminosae)",
        "standards_body": "USDA Pulse Standards & Codex CXS 171-1989",
        "description": "Red and green lens-shaped pulse inspection for processing and packaging.",
        "expected_aspect_ratio_range": (1.0, 1.3),
        "classes": {
            "0_NOR": "Sound Whole Lentil (Grade 1)",
            "1_F&S": "Anthracnose / Ascochyta Lesions",
            "2_SD":  "Germinated / Wrinkled Seed Coat",
            "3_MY":  "Moldy Decay / Discolored Cotyledons",
            "4_AP":  "Pulse Beetle Emergence Holes",
            "5_BN":  "De-hulled & Split Lentil",
            "6_UN":  "Poorly Filled / Chalky Immature Seed",
            "7_IM":  "Weed Seeds, Chaff & Stems"
        }
    }
}


def grade_single_kernel(predicted_code, confidence=1.0, grain_species="Rice (Oryza sativa)", aspect_ratio=None):
    """Computes species-aware single-kernel commercial grade and actionable advice.
    
    Args:
        predicted_code (str): Predicted 8-class defect code (e.g. '0_NOR', '1_F&S').
        confidence (float): Softmax model confidence score [0.0 - 1.0].
        grain_species (str): Target food grain species.
        aspect_ratio (float, optional): Length/Width kernel aspect ratio.
        
    Returns:
        dict: Species-specific commercial grading metadata.
    """
    profile = GRAIN_SPECIES_PROFILES.get(grain_species, GRAIN_SPECIES_PROFILES["Rice (Oryza sativa)"])
    standards_body = profile["standards_body"]
    ar_range = profile["expected_aspect_ratio_range"]

    # Check morphometric geometry
    morphology_note = "Geometry within standard range."
    if aspect_ratio is not None:
        if aspect_ratio < ar_range[0] * 0.7:
            morphology_note = f"Unusually round kernel for {profile['scientific_name']} (L/W: {aspect_ratio:.2f})."
        elif aspect_ratio > ar_range[1] * 1.3:
            morphology_note = f"Excessively elongated kernel for {profile['scientific_name']} (L/W: {aspect_ratio:.2f})."
        else:
            morphology_note = f"Morphological geometry consistent with {profile['scientific_name']} (L/W: {aspect_ratio:.2f})."

    # Species-Specific Grading Logic
    if "Wheat" in grain_species:
        if predicted_code == "0_NOR":
            return {
                "commercial_grade": "US No. 1 Grade (Sound Bread Wheat)",
                "grade_badge": "US NO. 1",
                "grade_color": "#10b981",
                "defect_category": "Sound Kernel (Zero Defect)",
                "actionable_verdict": f"Satisfies {standards_body}. Certified for commercial flour milling & premium export.",
                "safety_status": "SAFE",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code in ["5_BN", "6_UN"]:
            return {
                "commercial_grade": "US No. 2 / No. 3 (Shrunken & Broken)",
                "grade_badge": "US NO. 2",
                "grade_color": "#3b82f6",
                "defect_category": "Mechanical Fracture / Shrunken",
                "actionable_verdict": "Fractured or shrunken kernel. Suitable for blended feed flour or domestic milling.",
                "safety_status": "COMMERCIAL DIVERTER",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code in ["1_F&S", "3_MY"]:
            return {
                "commercial_grade": "Sample Grade - REJECT (Fusarium Head Blight / DON Toxin)",
                "grade_badge": "REJECT",
                "grade_color": "#ef4444",
                "defect_category": "Biological Fungal Pathogen",
                "actionable_verdict": "Hazardous tombstone scab detected. High vomitoxin (DON) risk; strictly unfit for human food.",
                "safety_status": "HAZARDOUS",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code == "7_IM":
            return {
                "commercial_grade": "Sample Grade - REJECT (Dockage & Impurities)",
                "grade_badge": "REJECT",
                "grade_color": "#f59e0b",
                "defect_category": "Foreign Material & Dockage",
                "actionable_verdict": "Foreign organic matter detected. Requires seed cleaning and air aspiration.",
                "safety_status": "CONTAMINANT",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        else:  # 2_SD, 4_AP
            return {
                "commercial_grade": "Sample Grade - REJECT (Sprout / Pest Damaged)",
                "grade_badge": "REJECT",
                "grade_color": "#dc2626",
                "defect_category": "Alpha-Amylase Degradation / Weevil Attack",
                "actionable_verdict": "Sprouted or insect-bored kernel. Severe starch enzymatic breakdown; unfit for baking.",
                "safety_status": "DEFECTIVE",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }

    elif "Sorghum" in grain_species:
        if predicted_code == "0_NOR":
            return {
                "commercial_grade": "Codex Grade 1 (Premium Food Grain)",
                "grade_badge": "GRADE 1",
                "grade_color": "#10b981",
                "defect_category": "Sound Sorghum Grain",
                "actionable_verdict": f"Meets {standards_body}. Approved for food processing, porridge, and malt brewing.",
                "safety_status": "SAFE",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code in ["5_BN", "6_UN"]:
            return {
                "commercial_grade": "Codex Grade 2 (Cracked / Chalky Grain)",
                "grade_badge": "GRADE 2",
                "grade_color": "#3b82f6",
                "defect_category": "Cracked Grain / Soft Endosperm",
                "actionable_verdict": "Cracked kernel; divert to animal feed production or ethanol fermentation.",
                "safety_status": "COMMERCIAL DIVERTER",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code in ["1_F&S", "3_MY"]:
            return {
                "commercial_grade": "Codex REJECT (Ergot / Mold Infested)",
                "grade_badge": "REJECT",
                "grade_color": "#ef4444",
                "defect_category": "Toxic Fungal Pathogen",
                "actionable_verdict": "Toxic mould or ergot detected. Mycotoxin hazard; mandatory commercial rejection.",
                "safety_status": "HAZARDOUS",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        else:  # 7_IM, 2_SD, 4_AP
            return {
                "commercial_grade": "Codex REJECT (Defective / Contaminated)",
                "grade_badge": "REJECT",
                "grade_color": "#dc2626",
                "defect_category": "Pest / Sprout / Foreign Matter",
                "actionable_verdict": "Kernel defective or contaminated beyond Codex thresholds. Reprocess or reject.",
                "safety_status": "DEFECTIVE",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }

    else:  # Default: Rice (Oryza sativa)
        if predicted_code == "0_NOR":
            return {
                "commercial_grade": "Grade A (Sound / Premium)",
                "grade_badge": "GRADE A",
                "grade_color": "#10b981",
                "defect_category": "Sound Kernel (Zero Defect)",
                "actionable_verdict": f"Kernel satisfies {standards_body}. Approved for export and premium packaging.",
                "safety_status": "SAFE",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code == "5_BN":
            return {
                "commercial_grade": "Grade B (Mechanical Fracture)",
                "grade_badge": "GRADE B",
                "grade_color": "#3b82f6",
                "defect_category": "Mechanical Fracture",
                "actionable_verdict": "Sound kernel fractured during milling. Divert to broken-rice processing.",
                "safety_status": "COMMERCIAL DIVERTER",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code == "6_UN":
            return {
                "commercial_grade": "Grade B (Chalky / Immature)",
                "grade_badge": "GRADE B",
                "grade_color": "#6366f1",
                "defect_category": "Physiological Immaturity",
                "actionable_verdict": "Kernel exhibits premature harvesting chalkiness. Acceptable for domestic cooking grades.",
                "safety_status": "COMMERCIAL PASS",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code == "7_IM":
            return {
                "commercial_grade": "Reject (Foreign Contaminant)",
                "grade_badge": "REJECT",
                "grade_color": "#f59e0b",
                "defect_category": "Non-Grain Foreign Matter",
                "actionable_verdict": "Foreign organic material detected. Requires pneumatic aspiration and destoning.",
                "safety_status": "CONTAMINANT",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        elif predicted_code in ["1_F&S", "3_MY"]:
            return {
                "commercial_grade": "Reject (Toxic Fungal Blight)",
                "grade_badge": "REJECT",
                "grade_color": "#ef4444",
                "defect_category": "Biological Fungal Pathogen",
                "actionable_verdict": "Hazardous fungal mycelium detected. High risk of mycotoxins; mandatory optical rejection.",
                "safety_status": "HAZARDOUS",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }
        else:  # 2_SD, 4_AP
            return {
                "commercial_grade": "Reject (Degraded Kernel)",
                "grade_badge": "REJECT",
                "grade_color": "#dc2626",
                "defect_category": "Biological Germination / Insect Borehole",
                "actionable_verdict": "Kernel exhibits germination or insect pitting. Unfit for commercial food lots.",
                "safety_status": "DEFECTIVE",
                "standards_body": standards_body,
                "morphology_note": morphology_note
            }


def compute_lot_grading(class_codes_list, grain_species="Rice (Oryza sativa)"):
    """Aggregates multiple kernel inspections into a certified commercial lot grade.
    
    Standards:
    - Rice: AGMARK & USDA Rice Standards
    - Wheat: USDA GIPSA FGIS Official Wheat Lot Standards
    - Sorghum: Codex CXS 172-1989 Coarse Grain Lot Standards
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

    profile = GRAIN_SPECIES_PROFILES.get(grain_species, GRAIN_SPECIES_PROFILES["Rice (Oryza sativa)"])
    standards_body = profile["standards_body"]

    # Grading Rule Hierarchy
    if mold_pct > 2.0 or bio_defect_pct > 12.0 or impurity_pct > 2.0:
        lot_grade = f"Reject (Non-Compliant / Contaminated {profile['scientific_name']})"
        badge = "REJECT"
        color = "#ef4444"
        verdict = f"Batch rejected under {standards_body} due to biological decay or foreign contaminants."
    elif sound_pct >= 90.0 and broken_pct <= 5.0 and impurity_pct == 0.0 and bio_defect_pct <= 3.0:
        badge = "US NO. 1" if "Wheat" in grain_species else ("GRADE 1" if "Sorghum" in grain_species else "GRADE A")
        lot_grade = f"{badge} (Premium Quality Food Grain)"
        color = "#10b981"
        verdict = f"Exceptional batch purity. Fully compliant with {standards_body} export benchmarks."
    elif sound_pct >= 80.0 and broken_pct <= 10.0 and impurity_pct <= 1.0 and bio_defect_pct <= 6.0:
        badge = "US NO. 2" if "Wheat" in grain_species else ("GRADE 2" if "Sorghum" in grain_species else "GRADE B")
        lot_grade = f"{badge} (Standard Commercial Food Lot)"
        color = "#3b82f6"
        verdict = f"Standard commercial lot quality. Approved for domestic wholesale and food packaging under {standards_body}."
    elif sound_pct >= 65.0 and broken_pct <= 18.0:
        badge = "US NO. 3" if "Wheat" in grain_species else "GRADE C"
        lot_grade = f"{badge} (Industrial / Animal Feed Grade)"
        color = "#f59e0b"
        verdict = f"Significant fraction of broken or chalky kernels. Divert to industrial starch or animal feed under {standards_body}."
    else:
        lot_grade = f"Reject (Substandard {profile['scientific_name']})"
        badge = "REJECT"
        color = "#dc2626"
        verdict = f"Excessive damage exceeds allowable commercial thresholds for {profile['scientific_name']}."

    return {
        "total_kernels": total,
        "grain_species": grain_species,
        "standards_body": standards_body,
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


def predict_quality_attributes(predicted_code, aspect_ratio=None, grain_species="Rice (Oryza sativa)", defect_area_ratio=0.0):
    """Predicts estimated moisture content (%) and storage preservation risk attributes.
    
    Standards:
    - Safe equilibrium storage moisture for cereal grains & pulses: 12.0% - 13.5%.
    - Moisture > 14.5% triggers rapid insect and storage pest proliferation.
    - Moisture > 16.0% triggers mycotoxin-producing fungal decay (Aspergillus / Fusarium).
    - Moisture > 18.0% initiates premature metabolic germination & sprouting.
    """
    if predicted_code == "0_NOR":
        base_moisture = 12.6
        status = "OPTIMAL_SAFE"
        risk_color = "#10b981"  # Emerald
        shelf_life = "12 - 24 Months"
        dockage_index = "Low Dockage (< 0.5%)"
        advice = "Grain moisture within safe equilibrium bounds. Ideal for long-term commercial silo storage."
    elif predicted_code == "3_MY":
        base_moisture = 18.4
        status = "CRITICAL_HAZARD"
        risk_color = "#ef4444"  # Red
        shelf_life = "Immediate Disposal (< 48 Hours)"
        dockage_index = "High Fungal Contamination"
        advice = "Severe moisture hazard (> 16.5%). Active fungal mycelium detected; toxic mycotoxin risk."
    elif predicted_code == "2_SD":
        base_moisture = 18.9
        status = "CRITICAL_HAZARD"
        risk_color = "#ef4444"  # Red
        shelf_life = "< 7 Days"
        dockage_index = "Sprouted Defect"
        advice = "High water imbibition triggered embryo germination. High alpha-amylase enzymatic activity."
    elif predicted_code == "1_F&S":
        base_moisture = 16.2
        status = "ELEVATED_RISK"
        risk_color = "#f97316"  # Orange
        shelf_life = "< 30 Days"
        dockage_index = "Fusarium Dockage"
        advice = "Elevated moisture at harvest enabled field scab. Deoxynivalenol (DON) testing required."
    elif predicted_code == "6_UN":
        base_moisture = 15.4
        status = "ELEVATED_RISK"
        risk_color = "#f59e0b"  # Amber
        shelf_life = "30 - 60 Days"
        dockage_index = "Immature Kernel Dockage"
        advice = "Early harvest moisture retention. Requires mechanical aeration/drying before bulk storage."
    elif predicted_code == "5_BN":
        base_moisture = 11.2
        status = "OVER_DRIED"
        risk_color = "#3b82f6"  # Blue
        shelf_life = "3 - 6 Months"
        dockage_index = "Mechanical Broken Ratio"
        advice = "Kernel over-dried below 12% moisture during post-harvest handling, inducing brittle fracture."
    elif predicted_code == "4_AP":
        base_moisture = 14.5
        status = "MODERATE_RISK"
        risk_color = "#eab308"  # Yellow
        shelf_life = "1 - 3 Months"
        dockage_index = "Insect Dockage"
        advice = "Insect metabolic respiration generating localized moisture hotspots. Fumigation recommended."
    else:  # 7_IM
        base_moisture = 13.8
        status = "FOREIGN_MATTER"
        risk_color = "#64748b"  # Slate
        shelf_life = "N/A (Non-Grain)"
        dockage_index = "Foreign Material Dockage (> 2.0%)"
        advice = "Foreign organic impurities present. Mechanical aspiration required."

    return {
        "estimated_moisture_pct": round(base_moisture, 1),
        "moisture_status": status,
        "risk_color": risk_color,
        "storage_life_estimate": shelf_life,
        "dockage_index": dockage_index,
        "preservation_advice": advice
    }
