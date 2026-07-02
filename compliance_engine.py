"""Sensitive data extraction, risk scoring, and LLM compliance analysis."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from config import settings
from prompts import COMPLIANCE_SUMMARY_PROMPT, CONTEXT_CLASSIFICATION_PROMPT
from utils import (
    counts_to_text,
    extract_text_from_csv,
    extract_text_from_pdf,
    extract_text_from_txt,
    get_file_extension,
    mask_value,
    truncate_text,
    validate_upload,
)


@dataclass(frozen=True)
class DetectionResult:
    """Structured output from document detection."""

    text: str
    masked_text: str
    findings: list[dict[str, Any]]
    counts: dict[str, int]
    risk_level: str
    risk_score: int
    risk_reason: str
    processing_time: float


class DataDetector:
    """Extract text, detect structured identifiers, mask values, and score risk."""

    PATTERNS: dict[str, re.Pattern[str]] = {
        "Aadhaar": re.compile(r"(?<!\d)(?:\d{4}[\s-]?\d{4}[\s-]?\d{4})(?!\d)"),
        "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        "Email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "Phone": re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
        "Credit Card": re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
        "Bank Account": re.compile(r"\b(?:account(?:\s+number)?|acct(?:\s+no\.?)?)\s*[:#-]?\s*(\d{9,18})\b", re.IGNORECASE),
        "API Key": re.compile(
            r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?",
            re.IGNORECASE,
        ),
        "Password": re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s,'\"]{6,})['\"]?", re.IGNORECASE),
        "Employee ID": re.compile(r"\b(?:EMP|Employee ID|EmployeeID)[\s:-]*[A-Z0-9]{3,12}\b", re.IGNORECASE),
    }

    def extract_text(self, filename: str, data: bytes) -> str:
        """Extract text from a supported uploaded document."""
        validate_upload(filename, len(data))
        extension = get_file_extension(filename)
        if extension == ".pdf":
            return extract_text_from_pdf(data)
        if extension == ".txt":
            return extract_text_from_txt(data)
        if extension == ".csv":
            return extract_text_from_csv(data)
        raise ValueError("Unsupported file extension.")

    def analyze(self, filename: str, data: bytes) -> DetectionResult:
        """Run extraction, detection, masking, and risk scoring."""
        started = time.perf_counter()
        text = self.extract_text(filename, data)
        if not text.strip():
            raise ValueError("The uploaded document does not contain readable text.")

        findings = self.detect(text)
        masked_text = self.mask_text(text, findings)
        counts = dict(Counter(item["category"] for item in findings))
        risk_level, risk_score, risk_reason = self.calculate_risk(counts)

        return DetectionResult(
            text=text,
            masked_text=masked_text,
            findings=findings,
            counts=counts,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_reason=risk_reason,
            processing_time=time.perf_counter() - started,
        )

    def detect(self, text: str) -> list[dict[str, Any]]:
        """Detect structured sensitive identifiers using compiled regex only."""
        findings: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()

        for category, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(1) if match.groups() else match.group(0)
                start = match.start(1) if match.groups() else match.start()
                end = match.end(1) if match.groups() else match.end()
                key = (category, start, end)
                if key in seen:
                    continue
                if category == "Credit Card" and not self._looks_like_credit_card(value):
                    continue
                seen.add(key)
                findings.append(
                    {
                        "category": category,
                        "value": value.strip(),
                        "masked_value": mask_value(category, value),
                        "start": start,
                        "end": end,
                        "confidence": "High",
                    }
                )

        return sorted(findings, key=lambda item: (item["start"], item["category"]))

    def mask_text(self, text: str, findings: list[dict[str, Any]]) -> str:
        """Return a masked copy of the original text."""
        masked = text
        for item in sorted(findings, key=lambda finding: finding["start"], reverse=True):
            masked = masked[: item["start"]] + item["masked_value"] + masked[item["end"] :]
        return masked

    def calculate_risk(self, counts: dict[str, int]) -> tuple[str, int, str]:
        """Calculate weighted risk score and level."""
        score = sum(settings.risk.weights.get(category, 1) * count for category, count in counts.items())
        total_items = sum(counts.values())
        critical_hits = [category for category in settings.risk.critical_categories if counts.get(category, 0) > 0]

        if critical_hits:
            return "High", score, f"Critical sensitive category detected: {', '.join(critical_hits)}."
        if score > settings.risk.high_threshold:
            return "High", score, f"Weighted risk score {score} exceeds {settings.risk.high_threshold}."
        if total_items > settings.risk.sensitive_count_threshold:
            return "High", score, f"Detected {total_items} sensitive items, above the threshold of {settings.risk.sensitive_count_threshold}."
        if settings.risk.medium_min <= score <= settings.risk.high_threshold:
            return "Medium", score, f"Weighted risk score {score} is in the medium range."
        return "Low", score, "Few or no sensitive identifiers were detected."

    @staticmethod
    def _looks_like_credit_card(value: str) -> bool:
        digits = [int(char) for char in value if char.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        checksum = 0
        parity = len(digits) % 2
        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0


class ComplianceLLM:
    """Generate compliance summaries and unstructured context signals using an LLM."""

    def __init__(self) -> None:
        self.provider = settings.llm.provider
        self._llm = None

    def available(self) -> bool:
        """Return whether a supported API key is configured."""
        return self.provider is not None

    def _client(self):
        if self._llm is not None:
            return self._llm
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=settings.llm.gemini_model,
                google_api_key=settings.llm.google_api_key,
                temperature=settings.llm.temperature,
            )
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=settings.llm.openai_model,
                api_key=settings.llm.openai_api_key,
                temperature=settings.llm.temperature,
            )
        else:
            raise RuntimeError("No LLM API key configured.")
        return self._llm

    def generate_summary(self, result: DetectionResult) -> str:
        """Generate a professional compliance report."""
        if not self.available():
            return self._fallback_summary(result)
        prompt = COMPLIANCE_SUMMARY_PROMPT.format(
            risk_level=result.risk_level,
            risk_score=result.risk_score,
            risk_reason=result.risk_reason,
            detection_counts=counts_to_text(result.counts),
            document_excerpt=truncate_text(result.masked_text, settings.llm.max_input_chars),
        )
        return self._invoke(prompt)

    def classify_context(self, text: str) -> str:
        """Use the LLM only for unstructured confidential context identification."""
        if not self.available():
            return "Confidential Business Information:\nNone identified\n\nTrade Secret:\nNone identified\n\nInternal Sensitive Context:\nNone identified"
        prompt = CONTEXT_CLASSIFICATION_PROMPT.format(document_excerpt=truncate_text(text, settings.llm.max_input_chars))
        return self._invoke(prompt)

    def _invoke(self, prompt: str) -> str:
        try:
            response = self._client().invoke(prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as exc:
            return f"LLM analysis could not be completed: {exc}"

    @staticmethod
    def _fallback_summary(result: DetectionResult) -> str:
        counts = counts_to_text(result.counts)
        return (
            "Executive Summary:\n"
            f"The document was classified as {result.risk_level} risk with a score of {result.risk_score}.\n\n"
            "Compliance Observations:\n"
            f"{counts}\n\n"
            "Security Risks:\n"
            f"{result.risk_reason}\n\n"
            "Suggested Remediation:\n"
            "Mask sensitive values, restrict access, rotate exposed secrets, and retain only necessary personal data."
        )
