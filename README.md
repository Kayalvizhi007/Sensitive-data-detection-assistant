# Sensitive Data Detection & Compliance Assistant

Production-inspired Streamlit application for detecting sensitive information in PDF, TXT, and CSV documents, classifying document risk, generating compliance summaries, masking detected values, answering document questions with RAG, and writing audit logs.

## Architecture

```text
Upload -> Text Extraction -> Regex Detection -> Risk Engine -> Masked Preview
                                  |
                                  v
                             Compliance LLM
                                  |
                                  v
Document Text -> Chunking -> Embeddings -> FAISS -> Retriever -> LLM Answer
```

## Folder Structure

```text
Sensitive-Data-Assistant/
|-- app.py
|-- compliance_engine.py
|-- rag_qa.py
|-- logger.py
|-- config.py
|-- prompts.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- assets/
`-- logs/
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Add a Gemini key in `.env`:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

OpenAI is also supported with `OPENAI_API_KEY` when Gemini is not configured.

## Detection Strategy

Structured identifiers are detected with compiled regular expressions, not an LLM:

- Aadhaar
- PAN
- Email
- Phone
- Credit Card
- Bank Account
- API Keys
- Passwords
- Employee IDs

The LLM is used only for compliance summaries and unstructured context review such as confidential business information, trade secrets, and internal sensitive context.

## Risk Engine

Weighted scoring:

- API Key, Password, Credit Card, Bank Account: 5
- Aadhaar: 4
- PAN: 3
- Employee ID: 2
- Phone, Email: 1

High risk is assigned for any API key, any credit card, score above 10, or more than five sensitive items. Medium risk is score 4 to 10. Low risk is below 4.

## RAG Workflow

The uploaded document is split with `RecursiveCharacterTextSplitter`, embedded using Gemini or OpenAI embeddings, indexed in FAISS, and queried through a retriever-backed LLM prompt. The vector database is stored in Streamlit session state so tab switching does not rebuild it.

## Bonus Features

- Retrieval-Augmented Generation
- Data Masking / Redaction
- Audit Logging

## Error Handling

The app handles unsupported formats, empty files, corrupted PDFs, missing API keys, LLM failures, embedding failures, and vector store failures with user-friendly Streamlit messages. Audit logs are written to `logs/audit_log.csv`.

## Challenges

Balancing deterministic detection with LLM assistance is important. Regex is reliable for structured identifiers, while the LLM is reserved for narrative compliance reasoning and contextual sensitivity.

## Future Improvements

- Add automated tests for regex detection and risk scoring.
- Add role-based access control around uploaded document review.
- Add secure encrypted storage for audit events in enterprise deployments.
