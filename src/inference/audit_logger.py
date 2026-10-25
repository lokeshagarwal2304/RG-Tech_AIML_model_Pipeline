"""
GrainSight AI: Production Inspection Audit Logger
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior ML Research Engineer

Logs every optical grain inspection transaction for enterprise compliance,
traceability, and regulatory quality assurance (ISO 9001 / AGMARK / USDA).
"""

import os
import csv
import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
AUDIT_LOG_FILE = os.path.join(LOGS_DIR, "inspection_audit.csv")

AUDIT_HEADERS = [
    "Timestamp",
    "Certificate_ID",
    "Grain_Species",
    "Class_Code",
    "Defect_Label",
    "Confidence_Pct",
    "Commercial_Grade",
    "Safety_Status",
    "Estimated_Moisture_Pct",
    "Storage_Risk",
    "Grain_Area_Px",
    "Defect_Area_Px",
    "Aspect_Ratio"
]


def init_audit_log():
    """Initializes the audit log file and directories if not present."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(AUDIT_HEADERS)


def record_inspection_audit(
    cert_id,
    grain_species,
    prediction_result,
    moisture_info,
    segmentation_info=None
):
    """Appends an enterprise inspection transaction record to the audit log."""
    init_audit_log()

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grading = prediction_result.get("grading", {})

    grain_px = segmentation_info.get("grain_pixel_area", 0) if segmentation_info else 0
    defect_px = segmentation_info.get("defect_pixel_area", 0) if segmentation_info else 0

    row = [
        now_str,
        cert_id,
        grain_species,
        prediction_result.get("predicted_code", "UNKNOWN"),
        prediction_result.get("human_label", "UNKNOWN"),
        f"{prediction_result.get('confidence_percentage', 0.0):.2f}",
        grading.get("grade_badge", "N/A"),
        grading.get("safety_status", "N/A"),
        f"{moisture_info.get('estimated_moisture_pct', 0.0):.1f}%",
        moisture_info.get("moisture_status", "N/A"),
        grain_px,
        defect_px,
        f"{prediction_result.get('aspect_ratio', 0.0):.2f}"
    ]

    with open(AUDIT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return AUDIT_LOG_FILE


def get_recent_audit_logs(limit=25):
    """Retrieves the most recent audit records for UI visualization."""
    init_audit_log()
    records = []
    if not os.path.exists(AUDIT_LOG_FILE):
        return records

    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    # Return latest records first
    return records[::-1][:limit]
