import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ):
        # Vector store
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build FAISS index
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # 🔁 Replace Groq with Local HF LLM
        self.llm = llm_model

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        # Retrieve from FAISS
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata") and r["metadata"].get("text")
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        # Strong summarization prompt (important for small models)
        prompt = f"""
You are an AI assistant.
Summarize the context strictly based on the information provided.

Context:
{context}

Query:
{query}

Summary:
"""

        return self.llm.invoke(prompt)
