"""Retrieval-Augmented Generation pipeline for document Q&A."""

from __future__ import annotations

from dataclasses import dataclass

from config import GOOGLE_API_KEY, settings
from prompts import RAG_QA_PROMPT


@dataclass
class RAGAnswer:
    """Answer and source snippets returned by the RAG pipeline."""

    answer: str
    sources: list[str]


class DocumentRAG:
    """Build and reuse an in-memory FAISS vector store for one document."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._vectorstore = None
        self._llm = None

    def build(self) -> None:
        """Create the vector database once."""
        if self._vectorstore is not None:
            return
        if not settings.llm.provider:
            raise RuntimeError("RAG requires GOOGLE_API_KEY/GEMINI_API_KEY or OPENAI_API_KEY.")

        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
        )
        chunks = splitter.create_documents([self.text])
        if not chunks:
            raise RuntimeError("No text chunks were available for retrieval.")
        self._vectorstore = FAISS.from_documents(chunks, self._embeddings())

    def ask(self, question: str) -> RAGAnswer:
        """Answer a question using retrieved document chunks."""
        if not question.strip():
            return RAGAnswer("Please enter a question about the uploaded document.", [])
        self.build()
        docs = self._vectorstore.similarity_search(question, k=settings.rag.retriever_k)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = RAG_QA_PROMPT.format(context=context, question=question.strip())
        try:
            response = self._chat().invoke(prompt)
            answer = getattr(response, "content", str(response)).strip()
        except Exception as exc:
            answer = f"The question could not be answered because the LLM call failed: {exc}"
        return RAGAnswer(answer=answer, sources=[doc.page_content for doc in docs])

    def _embeddings(self):
        if settings.llm.provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY or settings.llm.google_api_key,
            )
        if settings.llm.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(api_key=settings.llm.openai_api_key)
        raise RuntimeError("No embeddings provider configured.")

    def _chat(self):
        if self._llm is not None:
            return self._llm
        if settings.llm.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=settings.llm.gemini_model,
                google_api_key=GOOGLE_API_KEY or settings.llm.google_api_key,
                temperature=settings.llm.temperature,
            )
        elif settings.llm.provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=settings.llm.openai_model,
                api_key=settings.llm.openai_api_key,
                temperature=settings.llm.temperature,
            )
        else:
            raise RuntimeError("No LLM provider configured.")
        return self._llm
