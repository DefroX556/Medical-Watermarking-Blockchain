"""
Clinical Workflow Interface
----------------------------
Wraps watermark embedding/extraction into a clinical pipeline suitable for
hospital integration. Provides structured reports and radiologist feedback
collection.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from config import RESULTS_WATERMARKED, ensure_dirs
from utils import save_json, load_json, sha256_bytes

log = logging.getLogger(__name__)


class ClinicalPipeline:
    """
    High-level clinical workflow for medical image watermarking.

    Wraps the core embed/extract modules with clinical metadata,
    audit trail integration, and structured reporting.
    """

    def __init__(self, hospital_id: str = "HOSP-001", department: str = "Radiology") -> None:
        self.hospital_id = hospital_id
        self.department = department
        self._session_log: List[Dict[str, Any]] = []
        ensure_dirs()
        log.info("ClinicalPipeline initialized: hospital=%s, dept=%s",
                 hospital_id, department)

    def submit_image(
        self,
        image_path: str,
        patient_id: str,
        doctor_id: str,
        study_type: str = "general",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Submit a medical image for watermarking and blockchain registration.

        Parameters
        ----------
        image_path : str
            Path to the input medical image (DICOM, PNG, JPEG).
        patient_id : str
            Anonymized patient identifier.
        doctor_id : str
            Requesting physician identifier.
        study_type : str
            Type of study (e.g., 'mri', 'ct', 'xray').
        notes : str
            Clinical notes to associate with the image.

        Returns
        -------
        dict
            Complete submission result including watermark metadata,
            blockchain block, and clinical tracking ID.
        """
        from core.embed import embed_watermark

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate clinical tracking ID
        tracking_id = f"{self.hospital_id}-{int(time.time())}-{patient_id[-4:]}"

        # Watermark text includes clinical metadata
        wm_text = f"{self.hospital_id}:{patient_id}:{doctor_id}:{tracking_id}"

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(RESULTS_WATERMARKED, f"{base}_clinical.png")
        meta_path = os.path.join(RESULTS_WATERMARKED, f"{base}_clinical_meta.json")

        log.info("Clinical submission: tracking=%s, patient=%s, doctor=%s",
                 tracking_id, patient_id, doctor_id)

        # Embed watermark
        embed_result = embed_watermark(
            input_path=image_path,
            output_path=out_path,
            watermark_text=wm_text,
            metadata_path=meta_path,
        )

        # Build clinical record
        record = {
            "tracking_id": tracking_id,
            "hospital_id": self.hospital_id,
            "department": self.department,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "study_type": study_type,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input_image": image_path,
            "watermarked_image": out_path,
            "metadata_path": meta_path,
            "psnr": embed_result["psnr"],
            "ssim": embed_result["ssim"],
            "image_hash": embed_result["image_hash"],
            "blockchain_block": embed_result["block"],
            "status": "submitted",
        }

        self._session_log.append(record)

        # Save clinical record
        record_path = os.path.join(RESULTS_WATERMARKED, f"{base}_clinical_record.json")
        save_json(record_path, record)
        log.info("Clinical record saved: %s", record_path)

        return record

    def verify_image(self, image_path: str, metadata_path: str) -> Dict[str, Any]:
        """
        Verify a watermarked medical image against the blockchain.

        Returns a complete verification report.
        """
        from core.extract import extract_watermark

        log.info("Clinical verification: %s", image_path)

        extract_result = extract_watermark(image_path, metadata_path)

        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "image_path": image_path,
            "watermark_text": extract_result["watermark_text"],
            "blockchain_verified": extract_result["verified_in_blockchain"],
            "image_hash": extract_result["image_hash"],
            "retrieval_latency_ms": extract_result["retrieve_latency_ms"],
            "integrity_status": "INTACT" if extract_result["verified_in_blockchain"] else "COMPROMISED",
        }

        # Parse clinical info from watermark text
        parts = extract_result["watermark_text"].split(":")
        if len(parts) >= 4:
            report["parsed_hospital_id"] = parts[0]
            report["parsed_patient_id"] = parts[1]
            report["parsed_doctor_id"] = parts[2]
            report["parsed_tracking_id"] = ":".join(parts[3:])

        log.info("Verification result: %s", report["integrity_status"])
        return report

    def generate_report(self, image_hash: str) -> Dict[str, Any]:
        """
        Generate a clinical verification report for an image hash.
        Suitable for PDF rendering or audit submission.
        """
        from blockchain import find_all_by_image_hash

        blocks = find_all_by_image_hash(image_hash)

        report = {
            "report_type": "Clinical Verification Report",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "hospital_id": self.hospital_id,
            "department": self.department,
            "image_hash": image_hash,
            "blockchain_records": len(blocks),
            "blocks": blocks,
            "chain_of_custody": [],
        }

        for block in blocks:
            data = block.get("data", {})
            report["chain_of_custody"].append({
                "block_index": block["index"],
                "timestamp": block["timestamp"],
                "embed_timestamp": data.get("embed_timestamp"),
                "image_hash": data.get("image_hash", "")[:16] + "…",
            })

        log.info("Report generated for hash %s…: %d records",
                 image_hash[:12], len(blocks))
        return report

    def collect_feedback(
        self,
        tracking_id: str,
        radiologist_id: str,
        quality_rating: int,
        diagnostic_impact: str = "none",
        comments: str = "",
    ) -> Dict[str, Any]:
        """
        Collect radiologist feedback on watermarked image quality.

        Parameters
        ----------
        tracking_id : str
            Clinical tracking ID from submit_image.
        radiologist_id : str
            Identifier of the reviewing radiologist.
        quality_rating : int
            1-5 rating (5 = no perceptible quality loss).
        diagnostic_impact : str
            'none', 'minor', 'moderate', 'severe'.
        comments : str
            Free-text feedback.
        """
        feedback = {
            "tracking_id": tracking_id,
            "radiologist_id": radiologist_id,
            "quality_rating": quality_rating,
            "diagnostic_impact": diagnostic_impact,
            "comments": comments,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        log.info("Feedback collected: tracking=%s, rating=%d, impact=%s",
                 tracking_id, quality_rating, diagnostic_impact)
        return feedback

    def get_session_log(self) -> List[Dict[str, Any]]:
        """Return all records from this session."""
        return list(self._session_log)
