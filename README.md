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

1. Initialize and Activate Virtual Environment
Bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

2. Install Dependencies
Bash
pip install -r requirements.txt

4. Environment Configuration
Create a .env file in your root folder and add your API key configuration:

Plaintext
GOOGLE_API_KEY=your_google_gemini_api_key_here
# OpenAI is fully supported as an alternative runtime fallback option:
# OPENAI_API_KEY=your_openai_api_key_here

4. Run the Infrastructure
Bash
streamlit run app.py

## AI/ML Approach Used 
1. Hybrid Token Detection Strategy
    To maximize compute efficiency and maintain 100% processing reliability, structured identifiers are parsed deterministically via compiled regular expressions rather than using LLM tokens. The language model is completely insulated from basic text parsing and is reserved exclusively for complex semantic summaries and contextual risk analysis.

Regex Identifiers: Aadhaar, PAN, Email, Phone, Credit Card, Bank Account, API Keys, Passwords, Employee IDs.

LLM Analysis: The LLM is used only for compliance summaries and unstructured context reviews, such as corporate trade secrets, confidential business information, and internal sensitive contexts.

2. Weighted Compliance Risk Assessment Engine
Identified items flow into a mathematical risk scoring rubric based on categorical severity weights:

Weight 5 (Critical Exposure): API Key, Password, Credit Card, Bank Account

Weight 4 (High Exposure): Aadhaar

Weight 3 (Medium Exposure): PAN

Weight 2 (Low Exposure): Employee ID

Weight 1 (Informational): Phone, Email

Threshold Rules: High risk is assigned if any API key or credit card is found, if the aggregate score matches/exceeds 10, or if more than five sensitive items are isolated. Medium risk ranges from scores 4 to 10. Low risk is for any total below 4.

3. State-Optimized RAG Architecture
Chunking Engine: Extracted text objects are separated utilizing a RecursiveCharacterTextSplitter algorithm.

Embedding Layer: Vectors are modeled against specialized generative arrays using GoogleAIEmbeddings or OpenAIEmbeddings.

Indexing Core: Vector graphs are loaded into an in-memory FAISS layout cache hosted directly inside Streamlit's global state memory pool (st.session_state), preventing unnecessary reprocessing during user interactions.

## Challenges

- Balancing Deterministic Detection with LLM Autonomy: Relying entirely on LLMs to discover raw strings introduces execution overhead, random latency spikes, and tracking risks.

- Mitigation: Engineered a strict pipeline separation—employing optimized regex architectures for literal validations while limiting foundation model interactions to high-level compliance auditing, narrative generation, and document Q&A tracking.

## Future Improvements

**Automated Regression Frameworks**: Introduce comprehensive unit testing arrays (pytest) to validate regex stability and verify risk grading bounds across diverse data profiles.

**Role-Based Access Control (RBAC)**: Build standard security protocols to restrict document evaluation and system log privileges based on authenticated user roles.

**Encrypted Enterprise Event Vaults**: Transition native audit logging pipelines from basic CSV outputs into securely encrypted database storage engines optimized for high-volume enterprise environments.

## Working Prototype Deployment Link : 
👉 Live Application Link: https://sensitive-data-detection-assistant-y8mwwvwkbxvwpq7dqi25tn.streamlit.app/
  
