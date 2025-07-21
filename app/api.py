# app/api.py

# FastAPI backend

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from app.rag_pipeline import rag_qa_guarded


app = FastAPI(title="MedGuide AI - RAG API")

# Allowing local frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QARequest(BaseModel):
    question: str
    history: Optional[List[str]] = None 

# POST endpoint
@app.post("/chat")
async def chat_endpoint(req: QARequest):
    result = rag_qa_guarded(req.question)
    return {
        "answer": result["answer"],
        "score": result.get("score", None),
        "sources": result.get("sources", [])
    }