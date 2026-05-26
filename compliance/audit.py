"""
Immutable Audit Logger
-----------------------
Logs every embed/extract/verify operation with timestamp, user, action,
and image hash. Supports role-based access control stubs and data
retention policies.
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

from config import AUDIT_LOG_FILE, AUDIT_ENABLED, DATA_RETENTION_DAYS

log = logging.getLogger(__name__)


class Role(Enum):
    """Role-based access control roles."""
    ADMIN = "admin"
    DOCTOR = "doctor"
    RADIOLOGIST = "radiologist"
    AUDITOR = "auditor"
    TECHNICIAN = "technician"


class Permission(Enum):
    """Actions that can be controlled by RBAC."""
    EMBED = "embed"
    EXTRACT = "extract"
    VERIFY = "verify"
    VIEW_AUDIT = "view_audit"
    DELETE_AUDIT = "delete_audit"
    EXPORT_DATA = "export_data"


# Role → allowed permissions
ROLE_PERMISSIONS: Dict[Role, set] = {
    Role.ADMIN: {p for p in Permission},
    Role.DOCTOR: {Permission.EMBED, Permission.EXTRACT, Permission.VERIFY},
    Role.RADIOLOGIST: {Permission.EXTRACT, Permission.VERIFY, Permission.VIEW_AUDIT},
    Role.AUDITOR: {Permission.VIEW_AUDIT, Permission.EXPORT_DATA},
    Role.TECHNICIAN: {Permission.EMBED, Permission.VERIFY},
}


class AuditLogger:
    """
    Append-only audit logger for compliance tracking.

    Each entry is hashed to provide tamper evidence — if any entry is
    modified, subsequent hash chains break.
    """

    def __init__(self, log_path: str = AUDIT_LOG_FILE) -> None:
        self.log_path = log_path
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing audit log."""
        if os.path.isfile(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    self._entries = json.load(f)
                log.debug("Loaded %d audit entries.", len(self._entries))
            except (json.JSONDecodeError, IOError):
                self._entries = []
                log.warning("Audit log corrupted or unreadable — starting fresh.")

    def _save(self) -> None:
        """Persist audit log to disk."""
        tmp = self.log_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._entries, f, indent=2)
        os.replace(tmp, self.log_path)

    def _compute_entry_hash(self, entry: Dict[str, Any]) -> str:
        """Hash an entry for tamper evidence."""
        payload = json.dumps(
            {k: v for k, v in entry.items() if k != "entry_hash"},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def log_action(
        self,
        action: str,
        user_id: str,
        role: str,
        image_hash: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log an audit event.

        Parameters
        ----------
        action : str
            Action performed (e.g., 'embed', 'extract', 'verify').
        user_id : str
            Identifier of the user performing the action.
        role : str
            User's role (admin, doctor, radiologist, etc.).
        image_hash : str
            Hash of the image involved.
        details : dict, optional
            Additional context.
        """
        if not AUDIT_ENABLED:
            return {}

        prev_hash = self._entries[-1]["entry_hash"] if self._entries else "0"

        entry = {
            "index": len(self._entries),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch": time.time(),
            "action": action,
            "user_id": user_id,
            "role": role,
            "image_hash": image_hash,
            "details": details or {},
            "previous_hash": prev_hash,
        }
        entry["entry_hash"] = self._compute_entry_hash(entry)

        self._entries.append(entry)
        self._save()

        log.info("Audit: %s by %s (%s) on %s…",
                 action, user_id, role, image_hash[:12] if image_hash else "N/A")
        return entry

    def check_permission(self, role: str, permission: str) -> bool:
        """Check if a role has a specific permission."""
        try:
            r = Role(role)
            p = Permission(permission)
        except ValueError:
            log.warning("Unknown role '%s' or permission '%s'.", role, permission)
            return False
        return p in ROLE_PERMISSIONS.get(r, set())

    def get_entries(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Filter audit entries by user, action, or time."""
        results = self._entries
        if user_id:
            results = [e for e in results if e["user_id"] == user_id]
        if action:
            results = [e for e in results if e["action"] == action]
        if since:
            since_iso = since.isoformat()
            results = [e for e in results if e["timestamp"] >= since_iso]
        return results

    def verify_integrity(self) -> bool:
        """Verify the hash chain of all audit entries."""
        if not self._entries:
            return True

        for i, entry in enumerate(self._entries):
            expected = self._compute_entry_hash(entry)
            if entry["entry_hash"] != expected:
                log.error("Audit entry %d: hash mismatch.", i)
                return False
            if i > 0 and entry["previous_hash"] != self._entries[i - 1]["entry_hash"]:
                log.error("Audit entry %d: chain linkage broken.", i)
                return False

        log.info("Audit log integrity verified: %d entries OK.", len(self._entries))
        return True

    def enforce_retention(self) -> int:
        """
        Remove audit entries older than the configured retention period.

        Returns the number of entries removed.
        """
        cutoff = datetime.utcnow() - timedelta(days=DATA_RETENTION_DAYS)
        cutoff_iso = cutoff.isoformat()

        before_count = len(self._entries)
        self._entries = [e for e in self._entries if e["timestamp"] >= cutoff_iso]
        removed = before_count - len(self._entries)

        if removed > 0:
            self._save()
            log.info("Retention enforced: removed %d entries older than %s.",
                     removed, cutoff.date())
        return removed
