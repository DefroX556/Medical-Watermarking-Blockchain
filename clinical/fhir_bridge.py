"""
FHIR Bridge — HIS / EMR Integration
--------------------------------------
Constructs HL7 FHIR-compliant resources for healthcare system interoperability.

Supports:
- ImagingStudy resource creation
- DiagnosticReport attachment
- Patient reference linking
- Push/pull stubs for EMR endpoints

Reference: https://www.hl7.org/fhir/imagingstudy.html
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils import save_json

log = logging.getLogger(__name__)


class FHIRBridge:
    """
    Construct FHIR-compliant JSON resources for medical image watermarking
    metadata integration with Hospital Information Systems (HIS) and
    Electronic Medical Records (EMR).
    """

    FHIR_VERSION = "4.0.1"  # R4

    def __init__(
        self,
        base_url: str = "https://fhir.hospital.local/api",
        organization_id: str = "ORG-001",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.organization_id = organization_id
        log.info("FHIRBridge initialized: base_url=%s", self.base_url)

    @staticmethod
    def _generate_id() -> str:
        return str(uuid.uuid4())

    def create_imaging_study(
        self,
        patient_id: str,
        image_hash: str,
        watermark_metadata: Dict[str, Any],
        modality: str = "OT",       # OT = Other
        study_description: str = "Watermarked Medical Image",
    ) -> Dict[str, Any]:
        """
        Create a FHIR ImagingStudy resource linking watermark metadata.

        Parameters
        ----------
        patient_id : str
            FHIR Patient resource ID.
        image_hash : str
            SHA-256 hash of the watermarked image.
        watermark_metadata : dict
            Metadata from the embedding process.
        modality : str
            DICOM modality code (CT, MR, DX, US, OT, etc.).
        study_description : str
            Human-readable description.

        Returns
        -------
        dict
            FHIR ImagingStudy resource (JSON-serializable).
        """
        study_id = self._generate_id()

        resource = {
            "resourceType": "ImagingStudy",
            "id": study_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat() + "Z",
                "profile": ["http://hl7.org/fhir/StructureDefinition/ImagingStudy"],
            },
            "status": "available",
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "started": datetime.utcnow().isoformat() + "Z",
            "description": study_description,
            "numberOfSeries": 1,
            "numberOfInstances": 1,
            "series": [
                {
                    "uid": f"urn:oid:2.25.{uuid.uuid4().int}",
                    "modality": {
                        "system": "http://dicom.nema.org/resources/ontology/DCM",
                        "code": modality,
                    },
                    "description": "Watermarked image series",
                    "numberOfInstances": 1,
                    "instance": [
                        {
                            "uid": f"urn:oid:2.25.{uuid.uuid4().int}",
                            "sopClass": {
                                "system": "urn:ietf:rfc:3986",
                                "code": "urn:oid:1.2.840.10008.5.1.4.1.1.7",
                            },
                            "title": "Watermarked Instance",
                        }
                    ],
                }
            ],
            # Custom extension for watermark metadata
            "extension": [
                {
                    "url": "http://watermark.medical/fhir/extension/image-hash",
                    "valueString": image_hash,
                },
                {
                    "url": "http://watermark.medical/fhir/extension/watermark-psnr",
                    "valueDecimal": watermark_metadata.get("psnr", 0.0),
                },
                {
                    "url": "http://watermark.medical/fhir/extension/watermark-ssim",
                    "valueDecimal": watermark_metadata.get("ssim", 0.0),
                },
                {
                    "url": "http://watermark.medical/fhir/extension/blockchain-verified",
                    "valueBoolean": watermark_metadata.get("block") is not None,
                },
            ],
        }

        log.info("FHIR ImagingStudy created: id=%s, patient=%s, modality=%s",
                 study_id, patient_id, modality)
        return resource

    def create_diagnostic_report(
        self,
        patient_id: str,
        study_id: str,
        verification_result: Dict[str, Any],
        conclusion: str = "Image integrity verified via blockchain watermarking.",
    ) -> Dict[str, Any]:
        """
        Create a FHIR DiagnosticReport linked to an ImagingStudy.

        Parameters
        ----------
        patient_id : str
            FHIR Patient ID.
        study_id : str
            ImagingStudy resource ID.
        verification_result : dict
            Output from extract_watermark / verify_image.
        conclusion : str
            Report conclusion text.
        """
        report_id = self._generate_id()

        resource = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "meta": {
                "lastUpdated": datetime.utcnow().isoformat() + "Z",
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "RAD",
                            "display": "Radiology",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "18748-4",
                        "display": "Diagnostic imaging study",
                    }
                ],
                "text": "Watermark Integrity Verification",
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "issued": datetime.utcnow().isoformat() + "Z",
            "imagingStudy": [{"reference": f"ImagingStudy/{study_id}"}],
            "conclusion": conclusion,
            "extension": [
                {
                    "url": "http://watermark.medical/fhir/extension/blockchain-verified",
                    "valueBoolean": verification_result.get("verified_in_blockchain", False),
                },
                {
                    "url": "http://watermark.medical/fhir/extension/integrity-status",
                    "valueString": verification_result.get("integrity_status", "UNKNOWN"),
                },
            ],
        }

        log.info("FHIR DiagnosticReport created: id=%s, study=%s", report_id, study_id)
        return resource

    def create_patient_reference(
        self,
        patient_id: str,
        name: str = "Anonymized Patient",
    ) -> Dict[str, Any]:
        """Create a minimal FHIR Patient resource reference."""
        return {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {"lastUpdated": datetime.utcnow().isoformat() + "Z"},
            "active": True,
            "name": [{"use": "anonymous", "text": name}],
        }

    def push_to_emr(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stub: Push a FHIR resource to the EMR endpoint.

        In production, this would make an HTTP POST to the FHIR server.
        """
        resource_type = resource.get("resourceType", "Unknown")
        resource_id = resource.get("id", "unknown")
        endpoint = f"{self.base_url}/{resource_type}"

        log.info(
            "[STUB] EMR push: POST %s/%s (would send %d bytes)",
            endpoint, resource_id, len(str(resource)),
        )

        return {
            "status": "stub_success",
            "endpoint": endpoint,
            "resource_id": resource_id,
            "message": "EMR push stub — implement HTTP client for production.",
        }

    def pull_from_emr(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """
        Stub: Pull a FHIR resource from the EMR endpoint.
        """
        endpoint = f"{self.base_url}/{resource_type}/{resource_id}"
        log.info("[STUB] EMR pull: GET %s", endpoint)

        return {
            "status": "stub_success",
            "endpoint": endpoint,
            "message": "EMR pull stub — implement HTTP client for production.",
        }

    def export_bundle(
        self, resources: List[Dict[str, Any]], output_path: str,
    ) -> str:
        """
        Export a list of FHIR resources as a Bundle to a JSON file.
        """
        bundle = {
            "resourceType": "Bundle",
            "id": self._generate_id(),
            "type": "collection",
            "total": len(resources),
            "entry": [
                {
                    "fullUrl": f"urn:uuid:{r.get('id', self._generate_id())}",
                    "resource": r,
                }
                for r in resources
            ],
        }

        save_json(output_path, bundle)
        log.info("FHIR Bundle exported: %d resources → %s", len(resources), output_path)
        return output_path
