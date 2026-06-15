import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import requests

load_dotenv()


class RAGChatbot:
    """RAG chatbot — HuggingFace embeddings + Chroma retrieval + Anthropic generation."""

    def __init__(self):
        self.vectorstore  = None
        self.retriever    = None
        self.embeddings   = None
        self.chat_history = []
        self._initialize()

    def _initialize(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✓ Loaded HuggingFace embeddings")
        except Exception as e:
            print(f"⚠️ Error initializing embeddings: {e}")

    def load_documents(self, documents: list[str]):
        """Embed reviews and store in Chroma."""
        try:
            self.vectorstore = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                collection_name="business_reviews",
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )
            print(f"✓ Loaded {len(documents)} documents into vector store")
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            raise

    def _retrieve(self, query: str) -> list[str]:
        """Return page_content strings for the top-k relevant docs.
        Handles both LangChain >=0.2 (.invoke) and older (.get_relevant_documents)."""
        try:
            docs = self.retriever.invoke(query)
        except AttributeError:
            docs = self.retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]

    def _generate(self, query: str, context: str) -> str:
        """Generate an answer via the Anthropic Messages API."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return (
                f"**Relevant reviews found:**\n{context}"
            )

        system = (
            "You are a business sentiment analyst. "
            "Answer the user's question using ONLY the customer reviews provided. "
            "Be concise, cite specific examples, and if the reviews don't cover "
            "the topic say so clearly."
        )
        user_msg = f"Customer reviews:\n{context}\n\nQuestion: {query}"

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 512,
                "system": system,
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def get_response(self, query: str) -> str:
        try:
            if self.retriever is None:
                return "⚠️ No documents loaded. Please upload reviews first."

            snippets = self._retrieve(query)
            context  = "\n\n".join(f"- {s}" for s in snippets)
            return self._generate(query, context)

        except Exception as e:
            return f"Error generating response: {e}"