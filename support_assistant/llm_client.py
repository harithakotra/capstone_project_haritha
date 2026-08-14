"""
llm_client.py

Thin wrapper around a real LLM call, used ONLY by the optional MOCK_LLM=0
extension. Never imported when MOCK_LLM is at its default (unset or "1"),
so the graded baseline has no dependency on this file or on an API key.

Defaults to Groq's OpenAI-compatible free-tier API (console.groq.com).
Any other LLM API with a genuinely free tier is a drop-in substitute -
just change GROQ_BASE_URL / GROQ_MODEL / the env var read for the key below.
"""

import os

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"


def call_llm(system: str, user: str) -> str:
    """Call the configured free-tier LLM and return its raw text response.

    Requires the GROQ_API_KEY environment variable to be set (free signup at
    console.groq.com, no credit card required). This function is only ever
    invoked when MOCK_LLM=0 is explicitly set.
    """
    from openai import OpenAI  # Groq exposes an OpenAI-compatible endpoint

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 was set but GROQ_API_KEY is missing. Either unset "
            "MOCK_LLM (to use the graded mock baseline) or export a free "
            "Groq API key from console.groq.com."
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content
