from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

from ingest import ingest
from agent import ResearchAgent

app = FastAPI()


class Query(BaseModel):
    q: str


@app.post('/ingest')
async def ingest_endpoint(file: UploadFile = File(...)):
    saved = f"/tmp/{file.filename}"
    with open(saved, 'wb') as out:
        shutil.copyfileobj(file.file, out)
    ingest(saved)
    return {"status": "ok", "file": file.filename}


@app.post('/query')
async def query_endpoint(query: Query):
    agent = ResearchAgent()
    ans = agent.answer(query.q)
    return {"answer": ans}


@app.get('/status')
def status():
    return {"status": "ready"}
