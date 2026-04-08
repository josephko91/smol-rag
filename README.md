# smol-rag

A research assistant RAG (Retrieval-Augmented Generation) agent that takes a collection of PDFs, datasets, and other documents, then answers expert-level questions on the covered topics with evidence-backed responses.

## Project Stack

This project implements a production-ready RAG system with the following components:

1. **LLM Inference Layer**: [Ollama](https://ollama.ai) — local LLM server for running models
2. **LLM Model**: [Mistral 7B](https://mistral.ai) — quantized 7B-parameter model for local inference
3. **Embeddings**: [Nomic Embed Text](https://www.nomic.ai) — efficient embedding model for semantic search
4. **Vector Database**: [Chroma](https://www.trychroma.com) — vector storage and retrieval
5. **Framework Layer**: [LangChain](https://www.langchain.com) — orchestration of RAG pipelines

### Architecture

```
Documents (PDFs, CSVs, TXT)
    ↓
Ingest Pipeline (chunk, embed, persist)
    ↓
Chroma Vector DB (semantic indexing)
    ↓
LangChain RAG Chain (retrieval + generation)
    ↓
Ollama/Mistral 7B (local inference)
    ↓
Expert-level Answers
```

## Prerequisites

- **macOS** (M1/M2/Intel) or Linux
- **Ollama** installed and running locally
- **Python 3.9+**
- **Pip** or **Conda** for dependency management

## Installation

### 1. Install Ollama and Start Server

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral 7B model
ollama pull mistral:7b

# Start Ollama server (runs on http://localhost:11434)
ollama serve
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Ingest Documents

```bash
# Ingest a PDF or CSV file
python ingest.py --source ./path/to/document.pdf

# Ingest all documents in a directory
python ingest.py --source ./docs
```

### Query the Agent

#### Via Python API

```python
from agent import ResearchAgent

agent = ResearchAgent()
answer = agent.answer("What are the key findings in the provided documents?")
print(answer)
```

#### Via REST API

```bash
# Start API server
uvicorn api:app --reload --host 127.0.0.1 --port 8000

# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"q":"Explain the main conclusions"}'

# Ingest endpoint
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@./document.pdf"

# Check status
curl "http://localhost:8000/status"
```

## Configuration

Edit `config.py` to customize:

- `OLLAMA_BASE_URL`: Ollama server endpoint (default: http://localhost:11434)
- `MODEL_NAME`: LLM model name in Ollama (default: mistral:7b)
- `EMBEDDING_MODEL`: Embedding model (default: nomic-embed-text)
- `CHUNK_SIZE`: Document chunk size in characters (default: 2000)
- `TOP_K`: Number of retrieved chunks per query (default: 8)
- `MAX_PROMPT_TOKENS`: Max tokens in prompt context (default: 4000)

## Project Files

- `config.py` — Configuration and defaults
- `ingest.py` — Document ingestion, chunking, and embedding
- `retriever.py` — LangChain + Chroma retrieval logic
- `model_backends.py` — Ollama LLM backend
- `agent.py` — RAG chain and research agent
- `api.py` — FastAPI server for ingest/query
- `requirements.txt` — Python dependencies

## Performance Notes

- **Mistral 7B**: ~500ms—1s per query on M1/M2 Macs with Ollama
- **Memory**: ~4GB RAM for the model + indexing overhead
- **Throughput**: ~5—10 queries/second depending on hardware

## Troubleshooting

**Ollama connection error**: Ensure Ollama is running (`ollama serve`) on http://localhost:11434

**Model not found**: Pull the model first: `ollama pull mistral:7b`

**Embedding errors**: The system falls back to sentence-transformers if Nomic embeddings unavailable

## Roadmap

- [ ] Token-aware chunking with tiktoken
- [ ] Cross-encoder reranking for improved precision
- [ ] Hybrid BM25 + dense retrieval
- [ ] Local web UI
- [ ] Fine-tuning on domain-specific data

## License

MIT
