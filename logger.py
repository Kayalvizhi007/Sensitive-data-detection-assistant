"""Thread-safe CSV audit logging."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config import AUDIT_LOG_PATH


_LOCK = threading.Lock()
_COLUMNS = [
    "Timestamp",
    "Filename",
    "Risk Level",
    "Risk Score",
    "Detection Counts",
    "Processing Time",
    "Status",
]


@dataclass(frozen=True)
class AuditEvent:
    """Audit record for one document-processing attempt."""

    filename: str
    risk_level: str
    risk_score: int
    detection_counts: dict[str, int]
    processing_time: float
    status: str


class AuditLogger:
    """Append-only CSV logger guarded by a process-local lock."""

    def __init__(self, path=AUDIT_LOG_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        with _LOCK, self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
            writer.writeheader()

    def log(self, event: AuditEvent) -> None:
        """Append an audit event to disk."""
        row: dict[str, Any] = {
            "Timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "Filename": event.filename,
            "Risk Level": event.risk_level,
            "Risk Score": event.risk_score,
            "Detection Counts": json.dumps(event.detection_counts, sort_keys=True),
            "Processing Time": f"{event.processing_time:.3f}",
            "Status": event.status,
        }
        with _LOCK, self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
            writer.writerow(row)


audit_logger = AuditLogger()
