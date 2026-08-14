# Zepto Support Assistant (`/support_assistant`)

A small RAG-based customer support service for Zepto: an 8-document policy
corpus embedded into ChromaDB, a LangGraph-orchestrated intent router +
retriever, a Pydantic-enforced JSON response schema, and a FastAPI wrapper.

Graded baseline runs fully offline: **`MOCK_LLM` is left unset (defaults to
mock mode)**, so no signup, API key, or network call to any LLM provider is
needed. A real LLM call (Groq free tier) and a Hugging Face Spaces deployment
are both optional, ungraded extensions layered on top — see the toggles below.

## Setup

```bash
pip install -r requirements.txt
python ingest.py          # builds the ChromaDB collection from docs/
uvicorn main:app --host 0.0.0.0 --port 7860
```

The first run of `ingest.py` (or the first server startup) downloads the
`all-MiniLM-L6-v2` model from Hugging Face once (no account/API key needed)
and caches it locally; every embedding call after that runs on-device.

## Architecture: ingestion → embedding → retrieval → generation

**1. Ingestion — `ingest.py::load_documents()`**
Reads each of the 8 policy files in `docs/doc_01.txt` … `docs/doc_08.txt`.
Each file is treated as a single chunk (one policy topic per file, short
enough that no further splitting is needed). Each chunk keeps its source
document id (e.g. `doc_06`) as an identifier that flows through to the final
response's `sources` field.

**2. Embedding — `ingest.py::get_embedding_model()` / `build_collection()`**
Chunks are embedded locally with `sentence-transformers`
(`all-MiniLM-L6-v2`, no API key, runs on CPU) and written into a ChromaDB
`PersistentClient` collection named `zepto_policies` (cosine similarity
space), persisted to `chroma_db/` on disk. `main.py`'s startup hook calls
`get_collection()`, which builds this collection once and reuses it on
every request.

**3. Retrieval — `graph.py::retrieve_and_answer` node**
For queries classified as `policy_question`, this node embeds the incoming
query with the same model and runs `collection.query(..., n_results=3)`
against ChromaDB to fetch the top-3 most similar chunks. This step is real
in *both* MOCK_LLM modes — it needs no API key.

**4. Generation — `graph.py::retrieve_and_answer` / `direct_answer`,
prompt in `prompts.py`**
Only the final answer-writing step branches on `MOCK_LLM`:
- **Mock mode (default, graded)**: `retrieve_and_answer` returns
  `f"Based on the retrieved context: {top_chunk_snippet}"` built from the
  single top-ranked chunk (first ~200 chars), no LLM call.
  `direct_answer` (used for `general_question` queries, no retrieval
  step at all) returns the fixed string *"I can only answer questions
  about Zepto policies right now."*, also no LLM call.
- **Optional `MOCK_LLM=0` extension**: both nodes instead call a real LLM
  (`llm_client.py`, Groq free tier by default) — `retrieve_and_answer`
  uses the structured prompt in `prompts.py` (role/context/task/format/
  length skeleton, with a negative constraint and a few-shot example)
  grounded in the retrieved chunks; `direct_answer` prompts the LLM
  directly with no retrieval. Both validate the LLM's JSON output against
  the `AskResponse` Pydantic schema and retry up to 2 additional times
  with a corrective instruction if validation fails, falling back to a
  marked `[ERROR]` response if all attempts fail.

**Routing.** `graph.py::classify_intent` decides which generation path to
take: in mock mode via a keyword heuristic (checks for "delivery",
"return", "refund", "membership", "tracking", "cancel", "gift card",
"support hours"); in the optional extension, by asking the LLM to
classify. A conditional edge (`route_after_classify`) then sends the query
to `retrieve_and_answer` or `direct_answer` — this routing edge itself
never depends on `MOCK_LLM`, only the generation step inside each target
node does.

```
query
  │
  ▼
[classify_intent] ── keyword heuristic (mock) / LLM (MOCK_LLM=0)
  │
  ├─ policy_question ──▶ [retrieve_and_answer] ─▶ embed query ─▶ ChromaDB top-3
  │                          │
  │                          ├─ mock: canned "Based on the retrieved context: ..."
  │                          └─ MOCK_LLM=0: real LLM, grounded in retrieved chunks
  │
  └─ general_question ─▶ [direct_answer]
                             ├─ mock: fixed canned string
                             └─ MOCK_LLM=0: real LLM, no retrieval
  │
  ▼
AskResponse {answer, sources, confidence}   ◀── validated by schemas.py
```

**Schema enforcement — `schemas.py`.** Every path returns an `AskResponse`
(`answer: str`, `sources: list[str]`, `confidence: float` in `[0, 1]`). In
mock mode this is populated deterministically in code (no LLM output to
validate against); in the optional extension it's the real LLM's JSON
output, validated and retried as described above.

**API — `main.py`.** Wraps `graph.run_query()` behind `POST /ask`
(`AskRequest` → `AskResponse`), with a startup hook that pre-builds the
ChromaDB collection so the first request isn't slow.

## What changes between mock (default) and `MOCK_LLM=0`

| Stage | Mock (default, graded) | `MOCK_LLM=0` (optional) |
|---|---|---|
| `classify_intent` | keyword heuristic | LLM call |
| retrieval in `retrieve_and_answer` | real (ChromaDB) in both modes | same |
| answer generation | canned template string | real LLM + structured prompt + schema retry |
| `direct_answer` | fixed canned string | real LLM, no retrieval |
| Requires API key? | No | Yes (`GROQ_API_KEY`, free tier) |

## Example calls (captured with `MOCK_LLM` at its default)

**Call 1 — routed to `retrieve_and_answer` (policy keyword "delivery" present):**

Request:
```json
POST /ask
{"query": "How much does standard delivery cost?"}
```

Response:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_03", "doc_04"],
  "confidence": 1.0
}
```

**Call 2 — routed to `direct_answer` (no policy keyword present):**

Request:
```json
POST /ask
{"query": "What is the weather like today?"}
```

Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Two additional retrieval checks confirming chunks match the question asked:

- *"Can I get a refund for a damaged item?"* → top source `doc_06` (Damaged
  or Missing Items policy) ✅
- *"How do I cancel my order?"* → top source `doc_05` (Order Cancellation
  Policy) ✅

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# then:
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How much does standard delivery cost?"}'
```

The image builds and runs fully offline against the mock baseline
(`MOCK_LLM=1` is the image default). This local build/run is the required,
graded baseline — no push to Hugging Face Spaces is required.

### Optional extensions (not required for grading)

- **Real LLM (`MOCK_LLM=0`)**: export `GROQ_API_KEY` (free signup at
  console.groq.com, no credit card) and run with `-e MOCK_LLM=0 -e
  GROQ_API_KEY=...`. Not attempted for this submission — the graded mock
  baseline above is what was run and recorded.
- **Hugging Face Spaces deployment**: not attempted for this submission.
