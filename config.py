# Configuration defaults for smol-rag.
# LLM Inference via Ollama. Model selection can be overridden with
# environment variables to allow running quantized/Metal builds on Apple Silicon.
import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama API endpoint
# Default model can be overridden at runtime with the environment variable
# `MODEL_NAME`. Set this to your quantized/metal-enabled model if available.
MODEL_NAME = os.environ.get("MODEL_NAME", "mistral:7b")  # Model name as registered in Ollama
# Lower temperature (0.3) for more consistent, factual RAG answers (was 0.7)
# Set to 0.0 for fully deterministic output; higher values increase creativity/variability
MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", 0.3))
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", 256))

# Embeddings via Nomic
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")  # Nomic embed model (local or API)
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY")  # Set to use Nomic API; if None, assume local embedding service

# Vector Database (Chroma)
VECTOR_COLLECTION = "smol_rag_collection"
CHROMA_PERSIST_DIR = "./chroma_data"

# Chunking
CHUNK_SIZE = 2000  # characters per chunk (approx)
CHUNK_OVERLAP = 400

# Retrieval
TOP_K = 8
MAX_PROMPT_TOKENS = 4000
