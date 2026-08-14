"""
main.py

FastAPI wrapper around the LangGraph Support Assistant pipeline.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 7860

Endpoint:
    POST /ask   body: {"query": str}   ->  {"answer": str, "sources": [str], "confidence": float}

MOCK_LLM is left at its default (unset / "1") for grading - no API key or
network access to any LLM provider is required.
"""

from fastapi import FastAPI

from schemas import AskRequest, AskResponse
from graph import run_query
from ingest import get_collection

app = FastAPI(title="Zepto Support Assistant")


@app.on_event("startup")
def _warm_start():
    # Ensure the ChromaDB collection is built/loaded once at startup rather
    # than on the first request, so the first /ask call isn't slow.
    get_collection()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_query(request.query)


@app.get("/health")
def health():
    return {"status": "ok"}
