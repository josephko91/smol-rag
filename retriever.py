"""Retriever wrapper for Chromadb"""
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from config import EMBEDDING_MODEL, VECTOR_COLLECTION, TOP_K


class Retriever:
    def __init__(self):
        self.client = chromadb.Client(Settings())
        self.coll = self.client.get_collection(VECTOR_COLLECTION)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def retrieve(self, query: str, k: int = TOP_K):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        results = self.coll.query(query_embeddings=[q_emb], n_results=k)
        docs = []
        for d, m in zip(results['documents'][0], results['metadatas'][0]):
            docs.append({"text": d, "meta": m})
        return docs


if __name__ == '__main__':
    r = Retriever()
    print('Retriever ready')
