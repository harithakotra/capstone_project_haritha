FROM python:3.11-slim

WORKDIR /app

# System deps needed by torch / sentence-transformers at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docs/ ./docs/
COPY ingest.py prompts.py schemas.py graph.py llm_client.py main.py ./

# Pre-build the ChromaDB collection at image build time so the container
# starts up ready to serve (embedding model + corpus baked into the image).
RUN python ingest.py

# MOCK_LLM defaults to unset (mock mode) -- the graded baseline. Set
# MOCK_LLM=0 and GROQ_API_KEY as a runtime env var / secret to opt into the
# optional real-LLM extension; never bake an API key into the image.
ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
