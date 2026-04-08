"""Retriever wrapper using LangChain + Chromadb with Nomic embeddings"""
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, VECTOR_COLLECTION, TOP_K, CHROMA_PERSIST_DIR
import requests


def get_nomic_embedding_function():
    """Create embedding function for LangChain Chroma"""
    from langchain.embeddings.base import Embeddings
    from typing import List
    
    class NomicEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed search docs using Nomic or fallback to sentence-transformers"""
            if not texts:
                return []
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(EMBEDDING_MODEL)
                embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                raise RuntimeError(f"Embedding failed: {e}")

        def embed_query(self, text: str) -> List[float]:
            return self.embed_documents([text])[0]
    
    return NomicEmbeddings()


class Retriever:
    def __init__(self):
        persist_dir = CHROMA_PERSIST_DIR
        embeddings = get_nomic_embedding_function()
        
        self.retriever = Chroma(
            collection_name=VECTOR_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        ).as_retriever(search_kwargs={"k": TOP_K})

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        """Retrieve top-k documents for a query"""
        docs = self.retriever.invoke(query)
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "meta": getattr(doc, 'metadata', {})
            })
        return results


if __name__ == '__main__':
    r = Retriever()
    print('Retriever ready')
