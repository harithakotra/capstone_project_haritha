"""
graph.py

LangGraph-orchestrated flow for the Zepto Support Assistant.

State: TypedDict with the query, classified intent, retrieved chunks, and
final structured response.

Nodes (3):
  1. classify_intent    - routes each query to policy_question or general_question
  2. retrieve_and_answer - retrieval (always real) + answer generation (branches on MOCK_LLM)
  3. direct_answer       - answer generation with no retrieval (branches on MOCK_LLM)

Every node's *generation* step branches on the MOCK_LLM environment variable.
Retrieval inside retrieve_and_answer always runs for real in both modes,
since embedding + ChromaDB need no API key and no network call.
"""

import os
from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph, END

from ingest import get_collection, get_embedding_model
from schemas import AskResponse
from prompts import build_prompt

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

FALLBACK_GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."


def is_mock_mode() -> bool:
    """MOCK_LLM unset or '1' -> mock (graded baseline). MOCK_LLM='0' -> real LLM."""
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_chunks: list[dict]  # [{"id": str, "text": str}]
    response: AskResponse


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call the LLM to classify instead.
        intent = _llm_classify_intent(query)

    return {"intent": intent}


def _llm_classify_intent(query: str) -> str:
    """Optional real-LLM classification path (MOCK_LLM=0 only)."""
    from llm_client import call_llm  # imported lazily so mock mode never needs an LLM SDK

    system = (
        "Classify the customer query as exactly one word: either "
        "'policy_question' (about Zepto delivery, returns, membership, "
        "tracking, cancellation, gift cards, or support hours) or "
        "'general_question' (anything else). Respond with only that one word."
    )
    raw = call_llm(system=system, user=query).strip().lower()
    return "policy_question" if "policy_question" in raw else "general_question"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both modes (local embedding + ChromaDB,
    # no API key, no network call needed).
    collection = get_collection()
    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    retrieved_chunks = [
        {"id": doc_id, "text": text}
        for doc_id, text in zip(results["ids"][0], results["documents"][0])
    ]

    if is_mock_mode():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk = retrieved_chunks[0]
        top_chunk_snippet = top_chunk["text"][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        response = AskResponse(
            answer=answer,
            sources=[c["id"] for c in retrieved_chunks],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: prompt the real LLM, grounded in
        # the retrieved chunks, with schema validation + retry-on-failure.
        response = _llm_answer_with_retrieval(query, retrieved_chunks)

    return {"retrieved_chunks": retrieved_chunks, "response": response}


def _llm_answer_with_retrieval(query: str, retrieved_chunks: list[dict]) -> AskResponse:
    """Optional real-LLM generation path (MOCK_LLM=0 only), with schema
    validation and up to 2 corrective retries."""
    from llm_client import call_llm
    import json

    system, user = build_prompt(query, retrieved_chunks)

    last_error: Optional[Exception] = None
    for attempt in range(3):  # 1 initial attempt + 2 retries
        prompt_system = system
        if attempt > 0:
            prompt_system += (
                f"\n\nYour previous response failed schema validation with error: "
                f"{last_error}. Respond again with ONLY a valid JSON object "
                f"matching the required schema."
            )
        raw = call_llm(system=prompt_system, user=user)
        try:
            data = json.loads(raw)
            return AskResponse(**data)
        except Exception as e:  # JSON decode error or Pydantic ValidationError
            last_error = e
            continue

    return AskResponse(
        answer=f"[ERROR] Failed to produce a schema-valid response after retries: {last_error}",
        sources=[c["id"] for c in retrieved_chunks],
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        response = AskResponse(answer=FALLBACK_GENERAL_ANSWER, sources=[], confidence=1.0)
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        response = _llm_direct_answer(query)

    return {"retrieved_chunks": [], "response": response}


def _llm_direct_answer(query: str) -> AskResponse:
    from llm_client import call_llm
    import json

    system = (
        "You are Zepto's customer support assistant. The customer's question "
        "is unrelated to Zepto policy, so answer briefly and generally, then "
        "invite them to ask a Zepto-policy question instead. Respond with ONLY "
        'a JSON object: {"answer": str, "sources": [], "confidence": float}.'
    )
    last_error = None
    for attempt in range(3):
        prompt_system = system
        if attempt > 0:
            prompt_system += f"\n\nPrevious attempt failed validation: {last_error}. Return only valid JSON."
        raw = call_llm(system=prompt_system, user=query)
        try:
            data = json.loads(raw)
            return AskResponse(**data)
        except Exception as e:
            last_error = e
            continue
    return AskResponse(
        answer=f"[ERROR] Failed to produce a schema-valid response after retries: {last_error}",
        sources=[],
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def route_after_classify(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AskResponse:
    graph = get_graph()
    final_state = graph.invoke({"query": query})
    return final_state["response"]
