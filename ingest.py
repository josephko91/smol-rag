"""Ingest pipeline: PDFs and CSVs -> chunk -> embed with Nomic -> upsert to Chromadb"""
import argparse
import glob
import os
from typing import List

import chromadb
from chromadb.config import Settings
import pypdf
import pandas as pd
import requests

from config import EMBEDDING_MODEL, VECTOR_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, NOMIC_API_KEY, CHROMA_PERSIST_DIR


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using Nomic model via API or local service"""
    if not texts:
        return []
    
    # Try Nomic API if key provided
    if NOMIC_API_KEY:
        headers = {"Authorization": f"Bearer {NOMIC_API_KEY}"}
        payload = {"texts": texts, "model": EMBEDDING_MODEL}
        try:
            resp = requests.post("https://api.nomic.ai/embeddings", json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json().get("embeddings", [])
        except Exception as e:
            print(f"Nomic API error: {e}; falling back to sentence-transformers")
    
    # Fallback: use sentence-transformers locally (compatible with Nomic-like models)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


def read_pdf(path: str) -> str:
    try:
        reader = pypdf.PdfReader(path)
        out = []
        for p in reader.pages:
            out.append(p.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""


def read_csv(path: str) -> str:
    try:
        df = pd.read_csv(path)
        return df.to_csv(index=False)
    except Exception:
        return ""


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks


def ingest(source: str):
    # Create Chromadb client with persistence
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PERSIST_DIR,
        anonymized_telemetry=False
    )
    client = chromadb.Client(settings)
    coll = client.get_or_create_collection(VECTOR_COLLECTION)

    files = []
    if os.path.isdir(source):
        for ext in ("**/*.pdf", "**/*.csv", "**/*.txt"):
            files.extend(glob.glob(os.path.join(source, ext), recursive=True))
    elif os.path.isfile(source):
        files = [source]
    else:
        raise SystemExit("Source path not found")

    ids = []
    metadatas = []
    documents = []
    for f in files:
        text = ""
        if f.lower().endswith('.pdf'):
            text = read_pdf(f)
        elif f.lower().endswith('.csv'):
            text = read_csv(f)
        else:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()

        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            doc_id = f + "::" + str(i)
            ids.append(doc_id)
            metadatas.append({"source": f, "chunk": i})
            documents.append(c)

    if not documents:
        print("No documents found to ingest")
        return

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embed_texts(documents)
    coll.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    client.persist()
    print(f"Ingested {len(documents)} chunks from {len(files)} files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='File or directory to ingest')
    args = parser.parse_args()
    ingest(args.source)


if __name__ == '__main__':
    main()
