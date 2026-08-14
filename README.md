A single repository containing three modules that together demonstrate an
end-to-end pipeline: scraping and structuring raw data, analyzing it, and
serving a GenAI application on top of it.

| Module | Folder | What it does |
|---|---|---|
| 1. Data Pipeline | [`/data_pipeline`](./data_pipeline) | Scrapes a book catalog, cleans it, and loads it into SQLite |
| 2. Analytics | [`/analytics`](./analytics) | Runs SQL queries against the pipeline's data and verifies results with pandas |
| 3. Support Assistant | [`/support_assistant`](./support_assistant) | A LangGraph + FastAPI RAG service answering Zepto policy questions |

Each module has its own README with setup instructions, example runs, and
(for Module 3) an architecture write-up. This top-level README explains how
the three fit together and gives a quick-start for the whole repo.

> **Note:** Module 1 and Module 2 folder/file names below reflect the shape
> of that work as discussed previously (scrape → clean → SQLite → SQL query
> → pandas verification). Adjust the paths/filenames in this README if your
> actual files are named differently.

---

## 1. Data Pipeline — `/data_pipeline`

Scrapes book catalog data (title, price, rating, availability, category,
etc.), cleans and normalizes it, and stores it in a local SQLite database
that the other modules build on.

**Typical structure:**
```
data_pipeline/
├── scrape.py          # scrapes the book catalog source
├── clean.py           # normalizes/cleans the scraped records
├── load_to_sqlite.py  # writes cleaned records into SQLite
├── books.db            # resulting SQLite database
└── README.md
```

**Typical run:**
```bash
cd data_pipeline
python scrape.py            # produces raw scraped data
python clean.py             # cleans/normalizes it
python load_to_sqlite.py    # loads it into books.db
```

**Output:** `books.db`, a SQLite database with one or more tables holding
the cleaned book catalog — the shared data source for Module 2.

---

## 2. Analytics — `/analytics`

Runs SQL queries against `books.db` (produced by Module 1) to answer
catalog-level questions (e.g. price distribution by category, rating
trends, availability breakdowns), then cross-checks the SQL results against
the same data loaded into pandas, to confirm the query logic and the raw
data agree.

**Typical structure:**
```
analytics/
├── queries.sql         # the SQL queries being verified
├── verify_with_pandas.py  # loads books.db into pandas and re-derives the same results
└── README.md
```

**Typical run:**
```bash
cd analytics
python verify_with_pandas.py    # runs the SQL queries and the pandas equivalent, prints both
```

**Output:** a side-by-side comparison (in the script's output or a notebook)
showing the SQL query results match the independently-computed pandas
results — the verification step for Module 1's pipeline.

---

## 3. Support Assistant — `/support_assistant`

A small, complete GenAI service: an 8-document Zepto policy corpus embedded
locally and stored in ChromaDB, a 3-node LangGraph flow that classifies each
query and retrieves grounded context, a Pydantic-enforced JSON response
schema, and a FastAPI wrapper. The entire graded pipeline runs offline via a
deterministic mock mode (`MOCK_LLM` unset/`1`) — no API key or network call
to any LLM provider is required. A real LLM call (Groq free tier) and a
Hugging Face Spaces deployment are optional, ungraded extensions.

**Structure:**
```
support_assistant/
├── docs/                # doc_01.txt ... doc_08.txt, the policy corpus
├── ingest.py             # chunk + embed (all-MiniLM-L6-v2) + store in ChromaDB
├── prompts.py            # structured prompt template (optional real-LLM path)
├── graph.py              # LangGraph: classify_intent / retrieve_and_answer / direct_answer
├── llm_client.py         # Groq free-tier client (optional real-LLM path only)
├── schemas.py            # Pydantic request/response models
├── main.py                # FastAPI app, POST /ask
├── Dockerfile
├── requirements.txt
└── README.md              # full architecture write-up + example call transcripts
```

**Run locally:**
```bash
cd support_assistant
pip install -r requirements.txt
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 7860
```

**Run with Docker:**
```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

See [`support_assistant/README.md`](./support_assistant/README.md) for the
full pipeline architecture description and recorded example `/ask` calls.

---

## How the modules connect

```
[1. Data Pipeline]  scrape → clean → SQLite (books.db)
        │
        ▼
[2. Analytics]      SQL queries against books.db, cross-checked with pandas
        │
        ▼
[3. Support Assistant]   independent GenAI service (own corpus, own DB —
                          ChromaDB, not books.db); demonstrates the RAG/LLM
                          skillset the first two modules build toward
```

Modules 1 and 2 share the same SQLite database and form a classic
scrape → clean → store → query → verify data engineering pipeline. Module 3
is a self-contained GenAI application (its own document corpus and vector
store) demonstrating retrieval-augmented generation, structured LLM output,
and API deployment — the complementary skillset to the data pipeline work.

## Tech stack across the repo

- **Data pipeline:** Python, `requests`/`BeautifulSoup` (or similar), SQLite
- **Analytics:** SQL, pandas
- **Support Assistant:** sentence-transformers, ChromaDB, LangGraph,
  FastAPI, Pydantic, Docker

## Repo-wide setup

Each module has its own `requirements.txt` (or dependencies noted in its
README) and can be run independently — there's no single shared virtual
environment required across all three.
