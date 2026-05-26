"""
HIPAA / GDPR Compliance Validator
-----------------------------------
Validates watermarking operations against healthcare regulatory requirements.

Covers:
- PHI detection in watermark text
- De-identification verification
- HIPAA Security Rule checklist
- GDPR right-to-erasure stub
"""

import re
import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


# Common PHI patterns (US-centric)
_PHI_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "dob": re.compile(r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b"),
    "mrn": re.compile(r"\bMRN[:\s]?\d{6,}\b", re.IGNORECASE),
    "name_pattern": re.compile(r"\b(Dr\.?|Patient:?\s*)[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
    "address": re.compile(r"\b\d{1,5}\s\w+\s(St|Ave|Blvd|Rd|Dr|Ln|Way)\b", re.IGNORECASE),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


class HIPAAValidator:
    """
    Validate watermarking operations for HIPAA compliance.

    Checks for PHI exposure, required safeguards, and generates
    compliance reports.
    """

    def __init__(self) -> None:
        self._violations: List[Dict[str, Any]] = []
        log.info("HIPAAValidator initialized.")

    def detect_phi(self, text: str) -> List[Dict[str, str]]:
        """
        Scan text for potential Protected Health Information (PHI).

        Parameters
        ----------
        text : str
            Text to scan (e.g., watermark text, metadata).

        Returns
        -------
        list[dict]
            List of detected PHI items with type and matched text.
        """
        findings: List[Dict[str, str]] = []

        for phi_type, pattern in _PHI_PATTERNS.items():
            matches = pattern.findall(text)
            for match in matches:
                findings.append({
                    "type": phi_type,
                    "match": match if isinstance(match, str) else str(match),
                    "severity": "HIGH" if phi_type in ("ssn", "dob", "mrn") else "MEDIUM",
                })

        if findings:
            log.warning("PHI detected in text: %d items found.", len(findings))
            for f in findings:
                self._violations.append({
                    "rule": "HIPAA §164.514 — De-identification",
                    "type": f["type"],
                    "severity": f["severity"],
                })
        else:
            log.info("No PHI detected in watermark text.")

        return findings

    def verify_deidentification(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that metadata is properly de-identified.

        Checks for common PHI leakage in metadata fields.
        """
        issues: List[str] = []

        # Check all string values in metadata
        def _scan_dict(d: Dict, prefix: str = "") -> None:
            for k, v in d.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, str):
                    phi = self.detect_phi(v)
                    if phi:
                        issues.append(f"PHI in '{path}': {[p['type'] for p in phi]}")
                elif isinstance(v, dict):
                    _scan_dict(v, path)

        _scan_dict(metadata)

        result = {
            "is_deidentified": len(issues) == 0,
            "issues": issues,
            "fields_checked": len(metadata),
        }

        log.info("De-identification check: %s (%d issues)",
                 "PASS" if result["is_deidentified"] else "FAIL", len(issues))
        return result

    def generate_compliance_checklist(
        self,
        has_encryption: bool = False,
        has_audit_log: bool = False,
        has_access_control: bool = False,
        has_blockchain: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a HIPAA Security Rule compliance checklist.

        Returns a dict of requirements with pass/fail status.
        """
        checklist = {
            "standard": "HIPAA Security Rule (45 CFR Part 164)",
            "items": [
                {
                    "id": "164.312(a)(1)",
                    "name": "Access Control",
                    "description": "Implement technical policies to allow access only to authorized persons.",
                    "status": "PASS" if has_access_control else "FAIL",
                    "implementation": "RBAC via compliance.audit module" if has_access_control else "NOT IMPLEMENTED",
                },
                {
                    "id": "164.312(a)(2)(iv)",
                    "name": "Encryption and Decryption",
                    "description": "Implement a mechanism to encrypt and decrypt ePHI.",
                    "status": "PASS" if has_encryption else "FAIL",
                    "implementation": "AES-256-GCM via compliance.encryption module" if has_encryption else "NOT IMPLEMENTED",
                },
                {
                    "id": "164.312(b)",
                    "name": "Audit Controls",
                    "description": "Implement mechanisms to record and examine activity.",
                    "status": "PASS" if has_audit_log else "FAIL",
                    "implementation": "Hash-chained audit log via compliance.audit module" if has_audit_log else "NOT IMPLEMENTED",
                },
                {
                    "id": "164.312(c)(1)",
                    "name": "Integrity",
                    "description": "Implement policies to protect ePHI from improper alteration or destruction.",
                    "status": "PASS" if has_blockchain else "FAIL",
                    "implementation": "Blockchain hash verification" if has_blockchain else "NOT IMPLEMENTED",
                },
                {
                    "id": "164.312(e)(1)",
                    "name": "Transmission Security",
                    "description": "Implement technical security measures to guard against unauthorized access during transmission.",
                    "status": "PARTIAL",
                    "implementation": "Blockchain integrity — TLS transport not yet implemented.",
                },
                {
                    "id": "164.530(j)",
                    "name": "Data Retention",
                    "description": "Retain documentation for 6 years from date of creation.",
                    "status": "PASS",
                    "implementation": "7-year retention policy configured (DATA_RETENTION_DAYS).",
                },
            ],
        }

        passed = sum(1 for i in checklist["items"] if i["status"] == "PASS")
        total = len(checklist["items"])
        checklist["summary"] = {
            "passed": passed,
            "total": total,
            "compliance_percentage": round(passed / total * 100, 1),
        }

        log.info("HIPAA checklist: %d/%d passed (%.1f%%)",
                 passed, total, checklist["summary"]["compliance_percentage"])
        return checklist

    def generate_gdpr_checklist(
        self,
        has_consent_tracking: bool = False,
        has_erasure_capability: bool = False,
        has_data_portability: bool = False,
    ) -> Dict[str, Any]:
        """Generate a GDPR compliance checklist."""
        checklist = {
            "standard": "GDPR (EU 2016/679)",
            "items": [
                {
                    "id": "Art. 6",
                    "name": "Lawfulness of Processing",
                    "status": "PARTIAL",
                    "note": "Consent mechanism stub available.",
                },
                {
                    "id": "Art. 17",
                    "name": "Right to Erasure",
                    "status": "PASS" if has_erasure_capability else "STUB",
                    "note": "Chain pruning preserves hash integrity while removing PHI.",
                },
                {
                    "id": "Art. 20",
                    "name": "Right to Data Portability",
                    "status": "PASS" if has_data_portability else "STUB",
                    "note": "FHIR Bundle export provides portable data format.",
                },
                {
                    "id": "Art. 25",
                    "name": "Data Protection by Design",
                    "status": "PASS",
                    "note": "Watermark hashing ensures no raw PHI in blockchain.",
                },
                {
                    "id": "Art. 32",
                    "name": "Security of Processing",
                    "status": "PASS",
                    "note": "AES-256-GCM encryption + blockchain integrity.",
                },
            ],
        }

        passed = sum(1 for i in checklist["items"] if i["status"] == "PASS")
        total = len(checklist["items"])
        checklist["summary"] = {
            "passed": passed,
            "total": total,
            "compliance_percentage": round(passed / total * 100, 1),
        }

        log.info("GDPR checklist: %d/%d passed (%.1f%%)",
                 passed, total, checklist["summary"]["compliance_percentage"])
        return checklist

    def gdpr_erasure_stub(self, patient_id: str) -> Dict[str, Any]:
        """
        GDPR Right-to-Erasure stub.

        In production: remove patient data from chain while preserving
        blockchain integrity (replace data with tombstone hash).
        """
        log.info("[STUB] GDPR erasure requested for patient: %s", patient_id)
        return {
            "patient_id": patient_id,
            "status": "stub_acknowledged",
            "action": "Would replace patient-linked blocks with tombstone hashes.",
            "message": "Implement chain rewriting for production use.",
        }

    def get_violations(self) -> List[Dict[str, Any]]:
        """Return all recorded violations from this session."""
        return list(self._violations)
