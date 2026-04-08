"""Ingest pipeline: PDFs and CSVs -> chunk -> embed with Nomic -> upsert to Chromadb"""
import argparse
import glob
import os
from typing import List

import chromadb
# from chromadb.config import Settings
import pypdf
import pandas as pd
import requests

from config import EMBEDDING_MODEL, VECTOR_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, NOMIC_API_KEY, CHROMA_PERSIST_DIR


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using local nomic-embed-text model (batched, retried, validated)."""
    if not texts:
        return []
    from config import OLLAMA_BASE_URL
    embed_url = f"{OLLAMA_BASE_URL}/api/embed"
    batch_size = 64
    timeout = 120
    max_retries = 2
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        attempt = 0
        batch_embs = None
        while attempt <= max_retries:
            try:
                resp = requests.post(embed_url, json={"model": "nomic-embed-text", "input": batch}, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                batch_embs = data.get("embeddings")
                # basic validation
                if not batch_embs or len(batch_embs) != len(batch):
                    raise RuntimeError(f"Embedding count mismatch: expected {len(batch)}, got {len(batch_embs) if batch_embs is not None else 'None'}")
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Ollama embedding failed on batch starting at {start}: {e}")
                wait = 2 ** attempt
                time.sleep(wait)
        embeddings.extend(batch_embs)

    return embeddings

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
    # settings = Settings(
    #     chroma_db_impl="duckdb+parquet",
    #     persist_directory=CHROMA_PERSIST_DIR,
    #     anonymized_telemetry=False
    # )
    # client = chromadb.Client(settings)
    # coll = client.get_or_create_collection(VECTOR_COLLECTION)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    coll = client.get_or_create_collection(name=VECTOR_COLLECTION)

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
            if not c:
                continue
            c = c.strip()
            # skip very short chunks (noise)
            if len(c) < 50:
                continue
            doc_id = f + "::" + str(i)
            ids.append(doc_id)
            metadatas.append({"source": f, "chunk": i})
            documents.append(c)

    if not documents:
        print("No documents found to ingest")
        return

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embed_texts(documents)
    if not embeddings or len(embeddings) != len(documents):
        raise RuntimeError(f"Embeddings length {len(embeddings) if embeddings is not None else 0} does not match documents {len(documents)}")
    coll.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    # client.persist()
    print(f"Ingested {len(documents)} chunks from {len(files)} files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='File or directory to ingest')
    args = parser.parse_args()
    ingest(args.source)


if __name__ == '__main__':
    main()
