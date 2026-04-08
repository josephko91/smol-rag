"""Retriever wrapper using LangChain + Chromadb with Nomic embeddings"""
from typing import List, Dict
import chromadb
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, VECTOR_COLLECTION, TOP_K, CHROMA_PERSIST_DIR
import requests


def get_nomic_embedding_function():
    from langchain.embeddings.base import Embeddings
    from typing import List
    from config import OLLAMA_BASE_URL
    import requests, time

    class NomicEmbeddings(Embeddings):
        def _embed_batch(self, texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            url = f"{OLLAMA_BASE_URL}/api/embed"
            batch_size = 64
            timeout = 60
            max_retries = 2
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                attempt = 0
                while attempt <= max_retries:
                    try:
                        r = requests.post(url, json={"model":"nomic-embed-text","input": batch}, timeout=timeout)
                        r.raise_for_status()
                        data = r.json()
                        batch_embs = data.get("embeddings")
                        if not batch_embs or len(batch_embs) != len(batch):
                            raise RuntimeError("embedding count mismatch")
                        embeddings.extend(batch_embs)
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt > max_retries:
                            raise RuntimeError(f"Ollama embedding failed on batch starting at {i}: {e}")
                        time.sleep(2 ** attempt)
            return embeddings

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._embed_batch(texts)

        def embed_query(self, text: str) -> List[float]:
            return self._embed_batch([text])[0]

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
