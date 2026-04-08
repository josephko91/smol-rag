# Configuration defaults for smol-rag
# LLM Inference via Ollama
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama API endpoint
MODEL_NAME = "mistral:7b"  # Model name as registered in Ollama
MODEL_TEMPERATURE = 0.7
MODEL_MAX_TOKENS = 512

# Embeddings via Nomic
EMBEDDING_MODEL = "nomic-embed-text"  # Nomic embed model (local or API)
NOMIC_API_KEY = None  # Set to use Nomic API; if None, assume local embedding service

# Vector Database (Chroma)
VECTOR_COLLECTION = "smol_rag_collection"
CHROMA_PERSIST_DIR = "./chroma_data"

# Chunking
CHUNK_SIZE = 2000  # characters per chunk (approx)
CHUNK_OVERLAP = 400

# Retrieval
TOP_K = 8
MAX_PROMPT_TOKENS = 4000
