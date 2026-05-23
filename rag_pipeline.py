"""
Production RAG Pipeline  —  Section 6 Implementation
=====================================================
Demonstrates: correct embeddings, robust retrieval, grounded prompt,
edge case handling, and bottleneck awareness.

Adapted from the Athene prep guide to run with synthetic insurance data.

MODES
-----
DEMO MODE    (default)  – uses local sentence-transformers; no API key needed.
PRODUCTION MODE         – set OPENAI_API_KEY env var to use OpenAI embeddings
                          and GPT-4o for generation.

Usage:
    python main.py               # demo mode
    OPENAI_API_KEY=sk-... python main.py   # production mode
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# RULE 1: The same model MUST be used at index time and query time.
EMBEDDING_MODEL   = "text-embedding-3-small"   # OpenAI — 1536 dims
EMBEDDING_DIM     = 1536                        # Validate after every embed call
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"         # sentence-transformers — 384 dims

LLM_MODEL         = "gpt-4o"
LLM_TEMPERATURE   = 0.0     # Deterministic — required for factual RAG
LLM_MAX_TOKENS    = 1000
REQUEST_TIMEOUT   = 30      # seconds — prevent hanging on slow LLM calls

CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64
MIN_CHUNK_CHARS   = 50      # Filter undersized chunks — they are noise

DEFAULT_TOP_K     = 5
MAX_CONTEXT_TOKENS = 3_000  # Stay within LLM context budget
# Note: Threshold differs by embedding model:
#   OpenAI cosine similarity  → 0.70 is a reliable cutoff
#   sentence-transformers L2  → LangChain normalizes; 0.20 works better
OPENAI_SCORE_THRESHOLD = 0.70
LOCAL_SCORE_THRESHOLD  = 0.20

# ── Mode detection ────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI     = bool(OPENAI_API_KEY)
MIN_SCORE_THRESHOLD = OPENAI_SCORE_THRESHOLD if USE_OPENAI else LOCAL_SCORE_THRESHOLD

if USE_OPENAI:
    logger.info("PRODUCTION MODE: using OpenAI embeddings + GPT-4o")
else:
    logger.info("DEMO MODE: using local sentence-transformers (no API key required)")

# ── Embeddings ─────────────────────────────────────────────────────────────────
# NOTE: instantiate once and reuse — not per-request.
# Instantiating per-request wastes memory and, for OpenAI, triggers extra overhead.

def _build_embedding_model():
    """
    Return the appropriate embedding model based on environment.

    Production  → OpenAIEmbeddings (text-embedding-3-small, 1536 dims)
    Demo        → HuggingFaceEmbeddings (all-MiniLM-L6-v2, 384 dims, local)

    Returns:
        Initialized LangChain Embeddings object.

    Raises:
        ImportError: If required packages are not installed.
    """
    if USE_OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # BOTTLENECK: First instantiation downloads ~90 MB model weights.
        # In production, pre-warm on startup or cache the model on disk.
        return HuggingFaceEmbeddings(
            model_name=LOCAL_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        )


embeddings = _build_embedding_model()


# ── Chunking ───────────────────────────────────────────────────────────────────
def load_and_chunk(documents: list) -> list:
    """
    Split LangChain Document objects into overlapping chunks for indexing.

    In production, documents come from PyPDFLoader, TextLoader, etc.:
        loader = PyPDFLoader("path/to/policy.pdf")
        documents = loader.load()
    In this practice version, documents come from synthetic_data.py.

    Args:
        documents: List of LangChain Document objects. Must be non-empty.

    Returns:
        List of chunked Document objects with undersized chunks filtered out.

    Raises:
        ValueError: If documents is empty or chunking produces no valid chunks.
    """
    if not documents:
        raise ValueError("documents list cannot be empty")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(documents)

    # Filter undersized chunks — they carry too little signal to be useful
    chunks = [c for c in chunks if len(c.page_content.strip()) > MIN_CHUNK_CHARS]

    if not chunks:
        raise ValueError("No valid chunks produced from documents")

    logger.info(
        f"Chunked {len(documents)} documents → {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


# ── Indexing ───────────────────────────────────────────────────────────────────
def build_index(documents: list):
    """
    Build a FAISS vector index from a list of LangChain Documents.

    BOTTLENECK: Embedding all chunks is the dominant cost at index time.
      • OpenAI: ~$0.00002/1K tokens — 1,000 chunks ≈ $0.01
      • sentence-transformers: free but CPU-bound (~1–5s for small corpora)
    In production: batch embed, cache index to disk, re-index incrementally.

    Args:
        documents: List of LangChain Document objects to embed and index.

    Returns:
        Initialized FAISS VectorStore, ready for similarity search.

    Raises:
        ValueError: If documents is empty or chunking fails.
        RuntimeError: If the embedding call fails.
    """
    from langchain_community.vectorstores import FAISS

    chunks = load_and_chunk(documents)

    logger.info(f"Embedding {len(chunks)} chunks and building FAISS index…")
    start = time.time()

    try:
        # BOTTLENECK: This call embeds all chunks — most expensive step at index time.
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise RuntimeError(f"FAISS indexing error: {e}") from e

    elapsed = time.time() - start
    logger.info(f"FAISS index built — {len(chunks)} vectors in {elapsed:.2f}s")
    return vectorstore


# ── Retrieval ──────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its relevance metadata."""
    content:  str
    score:    float
    source:   str
    chunk_id: str = ""


def retrieve(
    query:           str,
    vectorstore,
    top_k:           int   = DEFAULT_TOP_K,
    score_threshold: float = MIN_SCORE_THRESHOLD,
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks for a query with score-based filtering.

    BOTTLENECK: Vector similarity search — ~50–200ms depending on index
    size and k. For large indexes, use ANN (HNSW). Reduce k. Pre-filter
    by metadata (doc_type, date) before the vector search.

    Args:
        query:           User query string. Must be non-empty.
        vectorstore:     Initialized FAISS vector store.
        top_k:           Maximum number of chunks to return.
        score_threshold: Minimum relevance score to include (0–1 scale).

    Returns:
        List of RetrievedChunk objects sorted by score descending.
        Returns empty list if no chunks exceed the threshold.

    Raises:
        ValueError:   If query is empty or whitespace-only.
        RuntimeError: If the vector search itself fails.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string")

    try:
        # BOTTLENECK: ~50–200ms depending on index size and k.
        # similarity_search_with_relevance_scores normalizes to [0, 1]:
        #   1.0 = identical, 0.0 = completely dissimilar.
        results = vectorstore.similarity_search_with_relevance_scores(
            query=query.strip(),
            k=top_k,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise RuntimeError(f"Retrieval error: {e}") from e

    if not results:
        logger.warning(f"No chunks retrieved for query: {query[:80]!r}")
        return []

    # Filter by score threshold — discard low-confidence matches
    chunks = [
        RetrievedChunk(
            content=doc.page_content,
            score=float(score),
            source=doc.metadata.get("source", "unknown"),
            chunk_id=doc.metadata.get("chunk_id", ""),
        )
        for doc, score in results
        if float(score) >= score_threshold
    ]

    if not chunks:
        best_score = results[0][1] if results else 0.0
        logger.warning(
            f"All {len(results)} results below threshold {score_threshold:.2f}. "
            f"Best score was {best_score:.3f}."
        )
    else:
        logger.info(
            f"Retrieved {len(chunks)}/{len(results)} chunks above threshold "
            f"{score_threshold:.2f} for query: {query[:60]!r}"
        )
        for chunk in chunks:
            logger.debug(f"  score={chunk.score:.3f}  source={chunk.source}")

    return chunks


# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert assistant for Athene annuity and retirement services professionals.
Your role is to answer questions about annuity products, retirement income planning,
pension risk transfer, and related regulatory and tax topics.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the CONTEXT provided below.
2. If the answer is not in the context, respond with exactly:
   "I could not find this information in the available documents."
3. Do NOT use prior knowledge, assumptions, or information outside the context.
4. Always cite the source document for each claim: [Source: <document name>]
5. If the context is ambiguous or contradictory, acknowledge this explicitly.

RESPONSE FORMAT:
- Be concise and direct.
- Use bullet points for multi-part answers.
- Include source citations as: [Source: <document name>]
"""

USER_PROMPT_TEMPLATE = """\
CONTEXT:
{context}

---

QUESTION: {question}

Remember: Use ONLY the context above to answer. \
If the answer is not present, say so explicitly.\
"""


def build_prompt(
    question: str,
    chunks:   list[RetrievedChunk],
) -> tuple[str, str]:
    """
    Assemble a grounded system + user prompt pair from retrieved chunks.

    Handles the no-context case explicitly so the LLM is never left without
    clear instructions — the most common source of hallucination in RAG.

    COST: Context size is the primary driver of LLM token cost.
    A strict char_budget prevents unbounded context from blowing the context window.

    Args:
        question: User's question. Must be non-empty.
        chunks:   Retrieved chunks from retrieve(). May be an empty list.

    Returns:
        Tuple of (system_prompt, user_prompt) ready for the chat API.

    Raises:
        ValueError: If question is empty or whitespace-only.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not chunks:
        # No context retrieved — instruct LLM to acknowledge this explicitly.
        # Never pass an empty CONTEXT block silently; that invites hallucination.
        logger.warning(f"No context for question: {question[:80]!r}")
        user_prompt = (
            "CONTEXT: No relevant documents were found.\n\n"
            f"QUESTION: {question.strip()}\n\n"
            "Since no context is available, respond with the no-information message."
        )
        return SYSTEM_PROMPT, user_prompt

    # Assemble context with token budget
    # COST: Every character here becomes a paid input token — be strict.
    context_parts: list[str] = []
    total_chars = 0
    char_budget = MAX_CONTEXT_TOKENS * 4  # ~4 chars per token (rough heuristic)

    for chunk in chunks:
        if total_chars + len(chunk.content) > char_budget:
            logger.warning("Context budget reached — truncating retrieved chunks")
            break
        context_parts.append(f"[Source: {chunk.source}]\n{chunk.content}")
        total_chars += len(chunk.content)

    context = "\n\n---\n\n".join(context_parts)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip(),
    )

    return SYSTEM_PROMPT, user_prompt


# ── Generation ─────────────────────────────────────────────────────────────────
def generate(
    question: str,
    chunks:   list[RetrievedChunk],
    model:    str = LLM_MODEL,
) -> dict:
    """
    Generate a grounded answer using the LLM (or a mock in demo mode).

    BOTTLENECK: LLM generation is the dominant latency driver — 500ms–3s+.
    COST:       Input tokens (context) + output tokens are the cost.
    Optimizations: reduce context size, stream output, use gpt-4o-mini for
    simple queries, cache answers to repeated questions.

    Args:
        question: User's question.
        chunks:   Retrieved chunks for grounding.
        model:    LLM model identifier (e.g., 'gpt-4o', 'gpt-4o-mini').

    Returns:
        Dict with keys: answer, sources, model, tokens.

    Raises:
        RuntimeError: On API timeout or non-retryable API error.
    """
    system_prompt, user_prompt = build_prompt(question, chunks)

    if not USE_OPENAI:
        # DEMO MODE: LLM call is skipped; mock response is returned instead.
        return _mock_generate(question, chunks, system_prompt, user_prompt)

    import openai

    try:
        # BOTTLENECK: Primary latency and cost driver for every query.
        start = time.time()
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=REQUEST_TIMEOUT,
        )
        elapsed = time.time() - start

        answer = response.choices[0].message.content
        logger.info(
            f"LLM generated answer in {elapsed:.2f}s "
            f"({response.usage.total_tokens} tokens)"
        )

        return {
            "answer":  answer,
            "sources": list({c.source for c in chunks}),
            "model":   model,
            "tokens":  response.usage.total_tokens,
        }

    except openai.RateLimitError as e:
        logger.error(f"LLM rate limit exceeded: {e}")
        raise  # Caller should implement exponential backoff

    except openai.APITimeoutError:
        logger.error(f"LLM request timed out after {REQUEST_TIMEOUT}s")
        raise RuntimeError("LLM generation timed out")

    except openai.APIError as e:
        logger.error(f"LLM API error: {e}")
        raise RuntimeError(f"Generation failed: {e}") from e


def _mock_generate(
    question:      str,
    chunks:        list[RetrievedChunk],
    system_prompt: str,
    user_prompt:   str,
) -> dict:
    """
    Demo-mode substitute for a real LLM call.

    Prints the full prompt that would be sent to GPT-4o, then returns a
    structured response built from the top retrieved chunk.  This lets you
    verify the retrieval stage end-to-end without an API key.
    """
    if not chunks:
        answer = "I could not find this information in the available documents."
    else:
        top = chunks[0]
        # Truncate to ~300 chars to keep demo output readable
        snippet = top.content[:300].rstrip() + ("…" if len(top.content) > 300 else "")
        answer = (
            f"[DEMO — GPT-4o not called]\n\n"
            f"Top retrieved chunk (score={top.score:.3f}):\n\n"
            f"{snippet}\n\n"
            f"[Source: {top.source}]\n\n"
            f"Set OPENAI_API_KEY to enable full LLM generation."
        )

    return {
        "answer":        answer,
        "sources":       list({c.source for c in chunks}),
        "model":         "demo-mock",
        "tokens":        len(user_prompt) // 4,   # rough token estimate
        # Demo extras — useful for studying the prompt
        "_system_prompt": system_prompt,
        "_user_prompt":   user_prompt,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────
def answer_question(
    question:    str,
    vectorstore,
) -> dict:
    """
    Full RAG pipeline: retrieve → prompt → generate.

    This is the entry point for a single query.  It wires together
    retrieve() and generate() and enforces top-level input validation.

    Args:
        question:    User's question. Must be a non-empty string.
        vectorstore: Initialized FAISS vector store (from build_index()).

    Returns:
        Dict with keys: answer, sources, model, tokens.

    Raises:
        ValueError:   If question is empty or whitespace-only.
        RuntimeError: If retrieval or generation fails.
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string")

    logger.info(f"── Answering: {question[:80]!r}")

    # Step 1: Retrieve  (BOTTLENECK: vector search ~50–200ms)
    chunks = retrieve(question, vectorstore)

    # Step 2: Generate  (BOTTLENECK: LLM call 500ms–3s+)
    return generate(question, chunks)
