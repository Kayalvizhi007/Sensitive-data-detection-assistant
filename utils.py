"""Reusable helpers for file handling, masking, and display-safe data shaping."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pandas as pd
from PyPDF2 import PdfReader

from config import settings


class FileValidationError(ValueError):
    """Raised when an uploaded file cannot be processed safely."""


def get_file_extension(filename: str) -> str:
    """Return the lower-case suffix for a filename."""
    return Path(filename).suffix.lower()


def validate_upload(filename: str, size: int) -> None:
    """Validate an uploaded file's extension and size."""
    extension = get_file_extension(filename)
    if extension not in settings.files.allowed_extensions:
        allowed = ", ".join(settings.files.allowed_extensions)
        raise FileValidationError(f"Unsupported file type. Allowed types: {allowed}.")

    max_bytes = settings.files.max_upload_mb * 1024 * 1024
    if size > max_bytes:
        raise FileValidationError(f"File is too large. Maximum size is {settings.files.max_upload_mb} MB.")


def calculate_file_hash(data: bytes) -> str:
    """Return a stable hash used to avoid duplicate processing in session state."""
    return hashlib.sha256(data).hexdigest()


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # PyPDF2 raises several parser-specific exceptions.
        raise FileValidationError("The PDF could not be read. It may be corrupted or encrypted.") from exc
    return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_text_from_txt(data: bytes) -> str:
    """Decode text bytes using UTF-8 with a practical fallback."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def extract_text_from_csv(data: bytes) -> str:
    """Convert CSV content into a readable text representation."""
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise FileValidationError("The CSV could not be parsed.") from exc
    if df.empty:
        return ""
    return df.to_csv(index=False)


def truncate_text(text: str, limit: int) -> str:
    """Return text clipped to a character limit with a clear marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Content truncated for model context limit.]"


def mask_value(category: str, value: str) -> str:
    """Mask a detected sensitive value while preserving useful recognition hints."""
    cleaned = value.strip()
    if not cleaned:
        return cleaned

    if category == "Email" and "@" in cleaned:
        local, domain = cleaned.split("@", 1)
        visible = local[:2] if len(local) > 2 else local[:1]
        return f"{visible}{'*' * max(4, len(local) - len(visible))}@{domain}"

    if category == "PAN":
        return cleaned[:3] + "****" + cleaned[-3:] if len(cleaned) >= 10 else "*" * len(cleaned)

    digits = "".join(char for char in cleaned if char.isdigit())
    if category in {"Aadhaar", "Credit Card", "Bank Account", "Phone"} and len(digits) >= 4:
        return "X" * max(0, len(digits) - 4) + digits[-4:]

    if category in {"API Key", "Password"}:
        return cleaned[:3] + "*" * max(6, len(cleaned) - 6) + cleaned[-3:] if len(cleaned) > 8 else "*" * len(cleaned)

    if len(cleaned) <= 4:
        return "*" * len(cleaned)
    return cleaned[:2] + "*" * max(3, len(cleaned) - 4) + cleaned[-2:]


def findings_to_dataframe(findings: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a display-ready findings table."""
    columns = ["Category", "Value", "Masked Value", "Start", "End", "Confidence"]
    if not findings:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "Category": item["category"],
            "Value": item["value"],
            "Masked Value": item["masked_value"],
            "Start": item["start"],
            "End": item["end"],
            "Confidence": item.get("confidence", "High"),
        }
        for item in findings
    ]
    return pd.DataFrame(rows, columns=columns)


def counts_to_text(counts: dict[str, int]) -> str:
    """Format detection counts for prompts and logging."""
    if not counts:
        return "None"
    return "\n".join(f"- {category}: {count}" for category, count in sorted(counts.items()))
