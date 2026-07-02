"""Central configuration for the Sensitive Data Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"
LOGS_DIR: Final[Path] = BASE_DIR / "logs"
AUDIT_LOG_PATH: Final[Path] = LOGS_DIR / "audit_log.csv"

ASSETS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY: Final[str | None] = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings loaded from environment variables."""

    google_api_key: str | None = GOOGLE_API_KEY
    openai_api_key: str | None = OPENAI_API_KEY
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_input_chars: int = int(os.getenv("LLM_MAX_INPUT_CHARS", "12000"))

    @property
    def provider(self) -> str | None:
        """Return the selected LLM provider, preferring Gemini."""
        if self.google_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return None


@dataclass(frozen=True)
class FileConfig:
    """File upload constraints."""

    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".csv")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))


@dataclass(frozen=True)
class RiskConfig:
    """Weighted scoring rules for sensitive data findings."""

    weights: dict[str, int] = field(
        default_factory=lambda: {
            "API Key": 5,
            "Password": 5,
            "Credit Card": 5,
            "Bank Account": 5,
            "Aadhaar": 4,
            "PAN": 3,
            "Employee ID": 2,
            "Phone": 1,
            "Email": 1,
            "Confidential Business Information": 2,
            "Trade Secret": 3,
            "Internal Sensitive Context": 2,
        }
    )
    high_threshold: int = 10
    medium_min: int = 4
    sensitive_count_threshold: int = 5
    critical_categories: tuple[str, ...] = ("API Key", "Credit Card")


@dataclass(frozen=True)
class RAGConfig:
    """Retrieval configuration."""

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    retriever_k: int = int(os.getenv("RAG_RETRIEVER_K", "4"))


@dataclass(frozen=True)
class AppConfig:
    """Application settings exposed as a single immutable object."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    files: FileConfig = field(default_factory=FileConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)


settings = AppConfig()
