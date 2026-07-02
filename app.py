"""Streamlit multi-step security assistant for sensitive data compliance."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from compliance_engine import ComplianceLLM, DataDetector, DetectionResult
from config import GOOGLE_API_KEY, settings
from logger import AuditEvent, audit_logger
from rag_qa import DocumentRAG, RAGAnswer
from utils import calculate_file_hash, counts_to_text, findings_to_dataframe, validate_upload


st.set_page_config(
    page_title="Sensitive Data Detection & Compliance Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@dataclass(frozen=True)
class PlaybookSections:
    """Executive playbook sections rendered in the compliance panel."""

    observations: list[str]
    risks: list[str]
    remediation: list[str]


RISK_THEME: dict[str, dict[str, str]] = {
    "Low": {"class": "risk-low", "label": "Low Threat", "tone": "Emerald"},
    "Medium": {"class": "risk-medium", "label": "Medium Threat", "tone": "Amber"},
    "High": {"class": "risk-high", "label": "High Threat", "tone": "Crimson"},
    "Unknown": {"class": "risk-unknown", "label": "Unclassified", "tone": "Slate"},
}

def inject_styles() -> None:
    """Inject production dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --page: #f5f7fb;
            --panel: #ffffff;
            --soft: #f8fafc;
            --ink: #0f172a;
            --muted: #64748b;
            --line: #dbe4ef;
            --blue: #2563eb;
            --navy: #10233f;
            --emerald-bg: #ecfdf5;
            --emerald-border: #86efac;
            --emerald-ink: #047857;
            --amber-bg: #fffbeb;
            --amber-border: #f6c453;
            --amber-ink: #b45309;
            --crimson-bg: #fff1f2;
            --crimson-border: #fb7185;
            --crimson-ink: #be123c;
            --shadow: 0 18px 46px rgba(15, 23, 42, 0.10);
            --shadow-sm: 0 8px 24px rgba(15, 23, 42, 0.07);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 30rem),
                linear-gradient(180deg, #f8fbff 0%, var(--page) 100%);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 17px 18px;
            box-shadow: var(--shadow-sm);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 850;
        }

        .center-shell {
            max-width: 980px;
            margin: 0 auto;
        }

        .landing-title {
            text-align: center;
            margin: 0 0 10px 0;
            font-size: 2.35rem;
            font-weight: 880;
        }

        .landing-subtitle {
            max-width: 820px;
            text-align: center;
            margin: 0 auto 28px auto;
            color: var(--muted);
            line-height: 1.58;
            font-size: 1.02rem;
        }

        .upload-zone, .panel, .summary-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .upload-zone {
            padding: 28px 30px 30px 30px;
            margin: 10px auto 20px auto;
        }

        .upload-zone h2 {
            margin: 0 0 8px 0;
            text-align: center;
            font-size: 1.45rem;
        }

        .upload-zone p {
            color: var(--muted);
            text-align: center;
            margin: 0 0 20px 0;
        }

        .info-panel {
            background: #eef6ff;
            border: 1px solid #bfdbfe;
            border-radius: 16px;
            padding: 16px 18px;
            color: #1d4b7a;
            box-shadow: var(--shadow-sm);
            margin: 16px 0;
        }

        .workspace-hero {
            background: linear-gradient(135deg, #10233f 0%, #214878 58%, #2d678f 100%);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 26px 30px;
            color: #f8fafc;
            margin-bottom: 18px;
        }

        .workspace-hero h1 {
            color: #ffffff;
            font-size: 1.9rem;
            margin: 0 0 8px 0;
        }

        .workspace-hero p {
            color: #cfe1f6;
            margin: 0;
            line-height: 1.55;
        }

        .stepper {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 0 auto 20px auto;
        }

        .step-pill {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 7px 12px;
            background: #ffffff;
            color: #64748b;
            font-weight: 850;
            font-size: 0.78rem;
        }

        .step-pill.active {
            background: #eaf2ff;
            border-color: #93c5fd;
            color: #1d4ed8;
        }

        .panel {
            padding: 22px 24px;
            margin: 16px 0;
        }

        .panel h2 {
            margin: 0 0 12px 0;
            font-size: 1.18rem;
        }

        .risk-card {
            border-radius: 16px;
            padding: 15px 17px;
            border: 1px solid;
            box-shadow: var(--shadow-sm);
            font-weight: 900;
            min-height: 108px;
        }

        .risk-low {
            background: linear-gradient(135deg, var(--emerald-bg), #fbfffd);
            border-color: var(--emerald-border);
            color: var(--emerald-ink);
        }

        .risk-medium {
            background: linear-gradient(135deg, var(--amber-bg), #fff8dc);
            border-color: var(--amber-border);
            color: var(--amber-ink);
        }

        .risk-high {
            background: linear-gradient(135deg, var(--crimson-bg), #fff7f8);
            border-color: var(--crimson-border);
            color: var(--crimson-ink);
        }

        .risk-unknown {
            background: #f8fafc;
            border-color: #cbd5e1;
            color: #475569;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }

        .summary-card {
            padding: 17px 18px;
            min-height: 275px;
            box-shadow: var(--shadow-sm);
        }

        .summary-card h3 {
            margin: 0 0 12px 0;
            font-size: 1rem;
            color: #16385f;
        }

        .summary-card li {
            margin-bottom: 10px;
            color: #334155;
            line-height: 1.48;
        }

        .qa-shell {
            background: #071527;
            border: 1px solid #24405f;
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 22px;
            color: #dbeafe;
            min-height: 420px;
        }

        .terminal-label {
            color: #93c5fd;
            text-transform: uppercase;
            font-size: 0.78rem;
            font-weight: 900;
            margin-bottom: 12px;
        }

        .chat-user, .chat-assistant {
            border-radius: 16px;
            padding: 14px 16px;
            margin: 12px 0;
            border: 1px solid;
            line-height: 1.55;
        }

        .chat-user {
            background: #0f2a4a;
            border-color: #315d8f;
            color: #e0f2fe;
        }

        .chat-assistant {
            background: #f8fafc;
            border-color: #dbe4ef;
            color: #172033;
        }

        .source-panel {
            background: #0b1220;
            border: 1px solid #1f3352;
            border-radius: 14px;
            color: #dbeafe;
            padding: 14px;
            white-space: pre-wrap;
            max-height: 240px;
            overflow: auto;
            font-family: Consolas, "Courier New", monospace;
            font-size: 0.84rem;
        }

        @media (max-width: 900px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
            .landing-title {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Initialize routing and cached processing state."""
    defaults: dict[str, Any] = {
        "current_step": 0,
        "file_hash": None,
        "filename": None,
        "result": None,
        "summary": None,
        "playbook": None,
        "context_classification": None,
        "rag": None,
        "rag_engine": None,
        "vector_store": None,
        "rag_build_error": None,
        "processing_error": None,
        "qa_history": [],
        "uploader_visible": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def configure_background_keys() -> None:
    """Silently load permanent API keys from config, st.secrets, or environment."""
    google_key = GOOGLE_API_KEY or settings.llm.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_key:
        google_key = read_secret_value(("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["GEMINI_API_KEY"] = google_key
        object.__setattr__(settings.llm, "google_api_key", google_key)

    openai_key = settings.llm.openai_api_key or os.getenv("OPENAI_API_KEY") or read_secret_value(("OPENAI_API_KEY",))
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        object.__setattr__(settings.llm, "openai_api_key", openai_key)


def read_secret_value(keys: tuple[str, ...]) -> str | None:
    """Read a Streamlit secret key without surfacing missing-secret exceptions."""
    if not streamlit_secrets_file_exists():
        return None
    try:
        for key in keys:
            value = st.secrets.get(key)
            if value:
                return str(value)
    except Exception:
        return None
    return None


def streamlit_secrets_file_exists() -> bool:
    """Return whether Streamlit secrets files exist before touching st.secrets."""
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    return any(path.exists() for path in candidates)


def llm_available() -> bool:
    """Return whether any supported permanent LLM provider is configured."""
    return bool(GOOGLE_API_KEY or settings.llm.google_api_key or settings.llm.openai_api_key)


def escape_html(value: str) -> str:
    """Escape text before rendering inside custom HTML."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def go_to_step(step: int) -> None:
    """Navigate between wizard steps without rebuilding cached artifacts."""
    st.session_state.current_step = step


def reset_workspace() -> None:
    """Reset document-specific session state."""
    st.session_state.current_step = 0
    st.session_state.file_hash = None
    st.session_state.filename = None
    st.session_state.result = None
    st.session_state.summary = None
    st.session_state.playbook = None
    st.session_state.context_classification = None
    st.session_state.rag = None
    st.session_state.rag_engine = None
    st.session_state.vector_store = None
    st.session_state.rag_build_error = None
    st.session_state.processing_error = None
    st.session_state.qa_history = []
    st.session_state.uploader_visible = True


def render_stepper(active_step: int) -> None:
    """Render progressive route pills."""
    labels = ("0. Ingestion Hub", "1. Compliance Panel", "2. Interactive Q&A")
    pills = []
    for index, label in enumerate(labels):
        active = " active" if index == active_step else ""
        pills.append(f'<div class="step-pill{active}">{label}</div>')
    st.markdown(f'<div class="stepper">{"".join(pills)}</div>', unsafe_allow_html=True)


def process_upload(filename: str, data: bytes) -> None:
    """Run extraction, detection, summary generation, logging, and optional RAG setup once."""
    file_hash = calculate_file_hash(data)
    if st.session_state.file_hash == file_hash and st.session_state.result is not None:
        return

    detector = DataDetector()
    llm = ComplianceLLM()
    started = time.perf_counter()

    try:
        with st.status("Running secure document pipeline", expanded=True) as status:
            st.write("Validating file, extracting readable content, and preparing scan buffers.")
            result = detector.analyze(filename, data)
            st.write("Detecting PAN, emails, phones, credit cards, bank accounts, API keys, passwords, employee IDs, and semantic corporate-secret signals.")
            st.write("Generating compliance diagnostics from masked document context.")
            summary = llm.generate_summary(result)
            context = llm.classify_context(result.masked_text)
            playbook = build_playbook(result, summary, context)
            rag, rag_error = build_rag_engine(result.masked_text) if llm_available() else (None, None)
            status.update(label="Document processed successfully", state="complete", expanded=False)

        st.session_state.file_hash = file_hash
        st.session_state.filename = filename
        st.session_state.result = result
        st.session_state.summary = summary
        st.session_state.context_classification = context
        st.session_state.playbook = playbook
        st.session_state.rag = rag
        st.session_state.rag_engine = rag
        st.session_state.vector_store = getattr(rag, "_vectorstore", None) if rag is not None else None
        st.session_state.rag_build_error = rag_error
        st.session_state.processing_error = None
        st.session_state.qa_history = []
        st.session_state.uploader_visible = False

        audit_logger.log(
            AuditEvent(
                filename=filename,
                risk_level=result.risk_level,
                risk_score=result.risk_score,
                detection_counts=result.counts,
                processing_time=time.perf_counter() - started,
                status="Success",
            )
        )
    except Exception as exc:
        st.session_state.processing_error = str(exc)
        st.session_state.uploader_visible = True
        audit_logger.log(
            AuditEvent(
                filename=filename,
                risk_level="Unknown",
                risk_score=0,
                detection_counts={},
                processing_time=time.perf_counter() - started,
                status=f"Failure: {exc}",
            )
        )


def build_rag_engine(masked_text: str) -> tuple[DocumentRAG | None, str | None]:
    """Build the RAG vector index immediately and return the ready engine."""
    try:
        rag = DocumentRAG(masked_text)
        rag.build()
        return rag, None
    except Exception as exc:
        return None, str(exc)


def build_playbook(result: DetectionResult, summary: str, context: str) -> PlaybookSections:
    """Create deterministic, detailed executive summary blocks."""
    total = sum(result.counts.values())
    categories = ", ".join(sorted(result.counts)) if result.counts else "no structured sensitive identifiers"
    observations = [
        f"The document is classified as {result.risk_level} risk with a score of {result.risk_score}, using weighted scoring and critical-category override rules.",
        f"The scanner detected {total} sensitive item(s) across {categories}. Coverage includes PAN numbers, emails, phones, credit cards, bank accounts, API keys, passwords, employee IDs, and semantic corporate-secret indicators.",
        f"Audit-ready detection counts: {counts_to_text(result.counts)}. {summarize_context_review(context)}",
    ]
    risks = [
        f"Primary risk driver: {result.risk_reason} This may create privacy, credential, financial, or internal confidentiality exposure depending on document distribution.",
        "Unmasked identifiers can increase regulatory exposure if the document is shared outside controlled workflows or retained longer than business necessity requires.",
        derive_summary_risk(summary),
    ]
    remediation = [
        "Use the masked preview as the approved sharing artifact and limit original-document access to least-privilege reviewers only.",
        "Rotate exposed API keys, passwords, access tokens, or secret material immediately, then verify downstream systems no longer accept old credentials.",
        "Attach this playbook to the audit trail, assign remediation ownership, and re-scan the sanitized document before external or cross-team release.",
    ]
    return PlaybookSections(observations=observations, risks=risks, remediation=remediation)


def summarize_context_review(context: str) -> str:
    """Return a compact semantic context sentence."""
    compact = " ".join(context.split())
    if not compact or ("None identified" in compact and len(compact) < 180):
        return "No strong semantic corporate-secret indicators were identified."
    return f"Semantic context review noted: {compact[:260]}."


def derive_summary_risk(summary: str) -> str:
    """Extract a risk sentence from the LLM/fallback summary."""
    compact = " ".join(summary.split())
    match = re.search(r"Security Risks:\s*(.+?)(?:Suggested Remediation:|$)", compact, flags=re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()[:360]
    return "The AI compliance narrative is advisory; deterministic findings and risk score remain the operational source of truth."


def render_landing_view() -> None:
    """Render Step 0: Ingestion Hub."""
    render_stepper(0)
    st.markdown(
        """
        <div class="center-shell">
            <h1 class="landing-title">Sensitive Data Detection & Compliance Assistant</h1>
            <p class="landing-subtitle">
                Upload one enterprise document to run sensitive-data detection, masking, risk scoring,
                audit logging, compliance diagnostics, and retrieval-based Q&A.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.uploader_visible:
        st.markdown(
            """
            <div class="center-shell">
                <div class="upload-zone">
                    <h2>Document Upload Zone</h2>
                    <p>Supported formats: PDF, TXT, and CSV. Analysis pages remain hidden until processing succeeds.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a compliance document",
            type=["pdf", "txt", "csv"],
            label_visibility="collapsed",
        )
        if uploaded_file is None:
            st.markdown(
                """
                <div class="center-shell">
                    <div class="info-panel">
                        No file is loaded. Drop or browse for a supported document to activate the scanner.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        data = uploaded_file.getvalue()
        try:
            validate_upload(uploaded_file.name, uploaded_file.size)
        except Exception as exc:
            st.error(str(exc))
            return
        process_upload(uploaded_file.name, data)

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)
        st.markdown(
            """
            <div class="info-panel">
                Processing failed. Please upload a readable PDF, TXT, or CSV and try again.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if st.session_state.result is not None:
        st.success(f"{st.session_state.filename} processed successfully.")
        if st.button("Proceed to Security Analysis Panel ➔", type="primary", use_container_width=True):
            go_to_step(1)
            st.rerun()


def render_compliance_view() -> None:
    """Render Step 1: Compliance Panel."""
    result: DetectionResult | None = st.session_state.result
    playbook: PlaybookSections | None = st.session_state.playbook
    if result is None or playbook is None:
        go_to_step(0)
        st.rerun()

    render_stepper(1)
    st.markdown(
        f"""
        <div class="workspace-hero">
            <h1>Security Analysis Panel</h1>
            <p>Evaluation workspace for <strong>{escape_html(st.session_state.filename or "uploaded document")}</strong>.
            Cached session artifacts prevent extraction and vector setup from rebuilding during navigation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.15, 1, 1])
    with col1:
        render_risk_card(result)
    col2.metric("Score", result.risk_score)
    col3.metric("Processing Time", f"{result.processing_time:.2f}s")

    render_visual_analytics(result)

    st.markdown('<div class="panel"><h2>Executive Compliance Summary</h2><div class="summary-grid">', unsafe_allow_html=True)
    render_summary_card("1. Compliance Observations", playbook.observations)
    render_summary_card("2. Security Risks", playbook.risks)
    render_summary_card("3. Suggested Remediation Steps", playbook.remediation)
    st.markdown("</div></div>", unsafe_allow_html=True)

    render_findings_sheet(result)

    st.download_button(
        "Download Playbook",
        data=build_report(result, playbook),
        file_name=f"compliance_playbook_{safe_filename(st.session_state.filename or 'document')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    left, _, right = st.columns([1, 2, 1.35])
    with left:
        if st.button("Back to Upload", use_container_width=True):
            reset_workspace()
            st.rerun()
    with right:
        if st.button("Proceed to Interactive Q&A Engine ➔", type="primary", use_container_width=True):
            go_to_step(2)
            st.rerun()


def render_risk_card(result: DetectionResult) -> None:
    """Render color-coded threat card."""
    theme = RISK_THEME.get(result.risk_level, RISK_THEME["Unknown"])
    st.markdown(
        f"""
        <div class="risk-card {theme["class"]}">
            <div style="font-size:0.78rem;text-transform:uppercase;opacity:.78;">Threat Level</div>
            <div style="font-size:1.45rem;margin-top:5px;">{theme["label"]}</div>
            <div style="font-size:0.86rem;margin-top:5px;">{theme["tone"]} posture | Score {result.risk_score}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_visual_analytics(result: DetectionResult) -> None:
    """Render dynamic category breakdown analytics below the compliance metrics."""
    chart_df = build_category_breakdown_frame(result)
    st.markdown('<div class="panel"><h2>Visual Analytics: Data Category Breakdown</h2>', unsafe_allow_html=True)

    if chart_df.empty:
        st.info("No sensitive data categories were detected, so there is no category breakdown to visualize.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=7, height=22)
        .encode(
            x=alt.X("Occurrences:Q", title="Total Occurrences", axis=alt.Axis(grid=True, tickMinStep=1)),
            y=alt.Y("Category:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260)),
            color=alt.Color(
                "Severity:N",
                title="Signal Severity",
                scale=alt.Scale(
                    domain=["Critical", "Elevated", "Standard"],
                    range=["#be123c", "#b45309", "#1e3a8a"],
                ),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip("Occurrences:Q", title="Occurrences"),
                alt.Tooltip("Severity:N", title="Severity"),
            ],
        )
        .properties(height=max(220, 42 * len(chart_df)))
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#334155", titleColor="#64748b", gridColor="#e5edf7")
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def build_category_breakdown_frame(result: DetectionResult) -> pd.DataFrame:
    """Build the visualization frame from cached scan findings/counts in session state."""
    counts = dict(result.counts)
    if not counts and result.findings:
        for finding in result.findings:
            category = str(finding.get("category", "Unknown"))
            counts[category] = counts.get(category, 0) + 1

    rows = [
        {
            "Category": normalize_category_label(category),
            "Occurrences": int(count),
            "Severity": category_severity(category),
        }
        for category, count in counts.items()
        if int(count) > 0
    ]
    return pd.DataFrame(rows).sort_values("Occurrences", ascending=False) if rows else pd.DataFrame()


def normalize_category_label(category: str) -> str:
    """Return presentation labels for sensitive data categories."""
    labels = {
        "PAN": "PAN Numbers",
        "Email": "Emails",
        "Phone": "Phone Numbers",
        "Credit Card": "Credit Cards",
        "API Key": "API Keys",
        "Password": "Passwords",
        "Bank Account": "Bank Accounts",
        "Employee ID": "Employee IDs",
        "Aadhaar": "Aadhaar IDs",
    }
    return labels.get(category, category)


def category_severity(category: str) -> str:
    """Classify chart color severity by sensitive data category."""
    if category in {"API Key", "Password", "Credit Card", "Bank Account"}:
        return "Critical"
    if category in {"Aadhaar", "PAN", "Employee ID"}:
        return "Elevated"
    return "Standard"


def render_summary_card(title: str, bullets: list[str]) -> None:
    """Render a summary section card."""
    items = "".join(f"<li>{escape_html(item)}</li>" for item in bullets[:3])
    st.markdown(
        f"""
        <div class="summary-card">
            <h3>{escape_html(title)}</h3>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_findings_sheet(result: DetectionResult) -> None:
    """Render findings and category matrices."""
    st.markdown('<div class="panel"><h2>Findings Data Sheet</h2>', unsafe_allow_html=True)
    findings_df = findings_to_dataframe(result.findings)
    if findings_df.empty:
        st.success("No structured sensitive identifiers were detected.")
    else:
        categories = sorted(findings_df["Category"].unique().tolist())
        selected = st.multiselect("Filter finding category", categories, default=categories)
        filtered = findings_df[findings_df["Category"].isin(selected)] if selected else findings_df
        st.dataframe(
            style_findings(filtered),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Signal Category", width="medium"),
                "Value": st.column_config.TextColumn("Detected Value", width="large"),
                "Masked Value": st.column_config.TextColumn("Masked Value", width="medium"),
                "Start": st.column_config.NumberColumn("Start", width="small"),
                "End": st.column_config.NumberColumn("End", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            },
        )

    counts_df = pd.DataFrame(
        [{"Category": category, "Count": count} for category, count in sorted(result.counts.items())]
    )
    if not counts_df.empty:
        st.dataframe(counts_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def style_findings(df: pd.DataFrame) -> Any:
    """Highlight findings by sensitivity."""
    critical = {"API Key", "Password", "Credit Card", "Bank Account"}
    moderate = {"Aadhaar", "PAN", "Employee ID"}

    def highlight(row: pd.Series) -> list[str]:
        category = row.get("Category")
        if category in critical:
            return ["background-color: #fff1f2; color: #9f1239; font-weight: 700"] * len(row)
        if category in moderate:
            return ["background-color: #fffbeb; color: #92400e; font-weight: 650"] * len(row)
        return ["background-color: #f8fafc; color: #334155"] * len(row)

    return df.style.apply(highlight, axis=1)


def build_report(result: DetectionResult, playbook: PlaybookSections) -> str:
    """Build downloadable compliance playbook."""
    lines = [
        "Sensitive Data Detection & Compliance Assistant",
        "Compliance Playbook",
        "",
        f"Filename: {st.session_state.filename}",
        f"Risk Level: {result.risk_level}",
        f"Score: {result.risk_score}",
        f"Processing Time: {result.processing_time:.2f}s",
        f"Total Findings: {sum(result.counts.values())}",
        f"Risk Reason: {result.risk_reason}",
        "",
        "Detection Counts:",
        counts_to_text(result.counts),
        "",
        "1. Compliance Observations:",
        *[f"- {item}" for item in playbook.observations],
        "",
        "2. Security Risks:",
        *[f"- {item}" for item in playbook.risks],
        "",
        "3. Suggested Remediation Steps:",
        *[f"- {item}" for item in playbook.remediation],
    ]
    return "\n".join(lines)


def safe_filename(value: str) -> str:
    """Return filesystem-safe filename text."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def render_qa_view() -> None:
    """Render Step 2: Interactive Q&A Engine."""
    result: DetectionResult | None = st.session_state.result
    if result is None:
        go_to_step(0)
        st.rerun()

    render_stepper(2)
    st.markdown(
        """
        <div class="workspace-hero">
            <h1>Interactive Q&A Engine</h1>
            <p>Ask focused questions about the masked document through a conversational security workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.rag_engine is None and llm_available():
        rag, rag_error = build_rag_engine(result.masked_text)
        st.session_state.rag = rag
        st.session_state.rag_engine = rag
        st.session_state.vector_store = getattr(rag, "_vectorstore", None) if rag is not None else None
        st.session_state.rag_build_error = rag_error

    render_chat_interface(st.session_state.rag_engine)

    left, _, right = st.columns([1, 2, 1.15])
    with left:
        if st.button("Back to Compliance Panel", use_container_width=True):
            go_to_step(1)
            st.rerun()
    with right:
        if st.button("Start New Scan", type="primary", use_container_width=True):
            reset_workspace()
            st.rerun()


def render_chat_interface(rag: DocumentRAG | None) -> None:
    """Render free-form document Q&A without preset prompts."""
    st.markdown(
        """
        <div class="panel">
            <h2>Ask Anything About The Uploaded Document</h2>
            <p style="color:#64748b;margin-top:0;">
                Type a natural question about the document, findings, risk posture, sensitive data, or remediation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    question = st.text_input(
        "Ask a question about the uploaded document",
        placeholder="Example: What compliance risks are present in this document?",
    )
    if st.button("Run Query", type="primary", use_container_width=True) and question.strip():
        answer = answer_question(rag, question.strip())
        st.session_state.qa_history.append({"question": question.strip(), "answer": answer})
        st.rerun()

    if not st.session_state.qa_history:
        st.markdown(
            '<div class="chat-assistant">Ready. Choose a quick prompt or enter a custom compliance question.</div>',
            unsafe_allow_html=True,
        )
    for item in st.session_state.qa_history:
        render_exchange(item["question"], item["answer"])
    


def answer_question(rag: DocumentRAG | None, question: str) -> RAGAnswer:
    """Answer using RAG when available, with a deterministic document-state fallback."""
    if rag is None and st.session_state.get("rag_engine") is not None:
        rag = st.session_state.rag_engine
    if rag is None and llm_available() and st.session_state.result is not None:
        rag, rag_error = build_rag_engine(st.session_state.result.masked_text)
        st.session_state.rag = rag
        st.session_state.rag_engine = rag
        st.session_state.vector_store = getattr(rag, "_vectorstore", None) if rag is not None else None
        st.session_state.rag_build_error = rag_error

    if rag is not None:
        try:
            with st.spinner("Retrieving relevant document context and generating answer..."):
                return rag.ask(question)
        except Exception as exc:
            st.session_state.rag_build_error = str(exc)

    return deterministic_document_answer(question)


def deterministic_document_answer(question: str) -> RAGAnswer:
    """Answer document questions from cached scan results when RAG is unavailable."""
    result: DetectionResult | None = st.session_state.result
    playbook: PlaybookSections | None = st.session_state.playbook
    if result is None:
        return RAGAnswer("No document is currently loaded. Upload a document first, then ask your question.", [])

    lowered = question.lower()
    counts_text = counts_to_text(result.counts)
    total_findings = sum(result.counts.values())

    if any(term in lowered for term in ["risk", "compliance", "issue", "problem", "exposure"]):
        risks = playbook.risks if playbook else [result.risk_reason]
        answer = (
            f"The document is classified as {result.risk_level} risk with a score of {result.risk_score}. "
            f"Total sensitive findings: {total_findings}.\n\n"
            "Compliance risks found:\n"
            + "\n".join(f"- {item}" for item in risks)
        )
        return RAGAnswer(answer, [result.masked_text[:1600]])

    if any(term in lowered for term in ["sensitive", "pii", "data", "findings", "identifier", "email", "phone", "pan", "password", "api", "credit", "bank", "employee"]):
        answer = (
            f"I found {total_findings} sensitive item(s) in the document.\n\n"
            f"Detection counts:\n{counts_text}\n\n"
            "The findings table on the Compliance Panel contains the detected values and masked equivalents."
        )
        return RAGAnswer(answer, [result.masked_text[:1600]])

    if any(term in lowered for term in ["summary", "summarize", "about", "overview"]):
        excerpt = " ".join(result.masked_text.split())[:900]
        answer = (
            f"Document overview: this file was processed as {result.risk_level} risk with "
            f"{total_findings} sensitive finding(s). Key detected categories: "
            f"{', '.join(sorted(result.counts)) if result.counts else 'none'}.\n\n"
            f"Masked excerpt: {excerpt}"
        )
        return RAGAnswer(answer, [result.masked_text[:1600]])

    if any(term in lowered for term in ["remediate", "remediation", "fix", "action", "recommend", "step", "priority"]):
        remediation = playbook.remediation if playbook else [
            "Use the masked preview before sharing.",
            "Restrict original-document access.",
            "Rotate any exposed credentials.",
        ]
        answer = "Recommended remediation steps:\n" + "\n".join(f"- {item}" for item in remediation)
        return RAGAnswer(answer, [result.masked_text[:1600]])

    matched_sentences = find_relevant_sentences(question, result.masked_text)
    if matched_sentences:
        answer = "Relevant document context:\n" + "\n".join(f"- {sentence}" for sentence in matched_sentences)
        return RAGAnswer(answer, matched_sentences)

    answer = (
        "I can answer from the uploaded document scan. The document has "
        f"{total_findings} sensitive finding(s), a {result.risk_level} risk level, and a score of {result.risk_score}. "
        "Ask about risks, sensitive data, summary, findings, or remediation for more detail."
    )
    return RAGAnswer(answer, [result.masked_text[:1600]])


def find_relevant_sentences(question: str, text: str, limit: int = 4) -> list[str]:
    """Find simple keyword-overlap sentences from the masked document."""
    stop_words = {
        "the", "is", "are", "a", "an", "to", "of", "for", "in", "on", "and", "or",
        "what", "which", "who", "where", "when", "how", "tell", "about", "document",
    }
    terms = {
        token
        for token in re.findall(r"[A-Za-z0-9_@.-]+", question.lower())
        if len(token) > 2 and token not in stop_words
    }
    if not terms:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        compact = " ".join(sentence.split())
        if not compact:
            continue
        lower = compact.lower()
        score = sum(1 for term in terms if term in lower)
        if score:
            scored.append((score, compact[:600]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in scored[:limit]]


def render_exchange(question: str, answer: RAGAnswer) -> None:
    """Render one chat exchange."""
    st.markdown(
        f'<div class="chat-user"><strong>Analyst</strong><br>{escape_html(question)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="chat-assistant"><strong>Assistant</strong><br>{escape_html(answer.answer)}</div>',
        unsafe_allow_html=True,
    )
    if answer.sources:
        with st.expander("Retrieved Evidence", expanded=False):
            for index, source in enumerate(answer.sources, start=1):
                st.markdown(f"**Source {index}**")
                st.markdown(f'<div class="source-panel">{escape_html(source[:1600])}</div>', unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit application."""
    configure_background_keys()
    inject_styles()
    initialize_state()

    step = st.session_state.current_step
    if step == 0:
        render_landing_view()
    elif step == 1:
        render_compliance_view()
    elif step == 2:
        render_qa_view()
    else:
        go_to_step(0)
        st.rerun()


if __name__ == "__main__":
    main()
