"""FastAPI server for smol-rag research agent"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from ingest import ingest
from agent import ResearchAgent

app = FastAPI(title="smol-rag", description="Research assistant RAG agent")

# Allow CORS for local UIs (gradio/streamlit/static pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent globally
try:
    agent = ResearchAgent()
except Exception as e:
    print(f"Warning: Agent initialization failed: {e}")
    agent = None


class QueryRequest(BaseModel):
    q: str


class IngestResponse(BaseModel):
    status: str
    file: str
    message: str


class QueryResponse(BaseModel):
    query: str
    answer: str


@app.post('/ingest', response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    """Ingest a document (PDF, CSV, or TXT)"""
    try:
        saved_path = f"/tmp/{file.filename}"
        with open(saved_path, 'wb') as out:
            shutil.copyfileobj(file.file, out)
        ingest(saved_path)
        return {
            "status": "success",
            "file": file.filename,
            "message": f"Ingested {file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@app.post('/query', response_model=QueryResponse)
async def query_endpoint(query: QueryRequest):
    """Query the research agent"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check Ollama connection.")
    try:
        answer = agent.answer(query.q)
        return {"query": query.q, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


class ChatRequest(BaseModel):
    message: str
    # Optional: when false, the agent will not retrieve documents and will
    # answer using the base LLM only.
    use_rag: bool | None = None


class ChatResponse(BaseModel):
    reply: str


@app.post('/chat', response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Simple chat endpoint that returns agent reply for a single message."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check Ollama connection.")
    try:
        # Pass the optional use_rag flag through to the agent. If None, agent
        # will auto-decide whether to retrieve.
        reply = agent.answer(req.message, use_rag=req.use_rag)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post('/reset-conversation')
async def reset_conversation():
    """Clear conversation history to start a fresh conversation."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Check Ollama connection.")
    try:
        agent.clear_history()
        return {"status": "success", "message": "Conversation history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get('/status')
def status():
    """Check system status"""
    return {
        "status": "ready" if agent else "degraded",
        "agent_ready": agent is not None,
        "message": "Agent ready for queries" if agent else "Agent initialization pending"
    }


@app.get('/')
def root():
    return {"name": "smol-rag", "description": "Research assistant RAG agent"}
