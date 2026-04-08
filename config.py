# Configuration defaults for smol-rag
MODEL_NAME = "mistral-7b"  # placeholder name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_COLLECTION = "smol_rag_collection"
CHUNK_SIZE = 2000  # characters per chunk (approx)
CHUNK_OVERLAP = 400
TOP_K = 8
LLAMA_MODEL_PATH = "./models/mistral-7b.ggml"  # path for ggml model used by llama.cpp
