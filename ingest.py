"""Simple ingest script: PDFs and CSVs -> chunk -> embed -> upsert to Chromadb"""
import argparse
import glob
import os
from typing import List

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pypdf
import pandas as pd

from config import EMBEDDING_MODEL, VECTOR_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP


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


def ingest(source: str, persist_directory: str = None):
    # create embedding model
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.Client(Settings())
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

    embeddings = embedder.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    coll.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings.tolist())
    print(f"Ingested {len(documents)} chunks from {len(files)} files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='File or directory to ingest')
    args = parser.parse_args()
    ingest(args.source)


if __name__ == '__main__':
    main()
