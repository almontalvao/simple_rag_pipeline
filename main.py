"""
main.py — RAG Pipeline Demo Runner
===================================
Builds a FAISS index from synthetic insurance documents, then runs a suite
of representative questions through the full pipeline.

Run (demo — no API key needed):
    python main.py

Run (production — uses OpenAI):
    OPENAI_API_KEY=sk-... python main.py
"""

import logging
import sys

from synthetic_data import get_synthetic_documents
from rag_pipeline import build_index, answer_question

logger = logging.getLogger(__name__)

# ── Demo queries ──────────────────────────────────────────────────────────────
# Span the full document corpus so every retrieval path gets exercised.
DEMO_QUESTIONS = [
    # FIA fundamentals
    "How does interest get credited in a Fixed Indexed Annuity?",
    # MYGA vs CD
    "How does a MYGA differ from a bank CD, and what are the tax advantages?",
    # GLWB rider mechanics
    "How is the Guaranteed Lifetime Withdrawal Benefit (GLWB) Benefit Base different from the account value?",
    # Surrender charges and liquidity
    "What surrender charge waivers are available on Athene annuity contracts?",
    # Tax treatment
    "Are non-qualified annuities subject to Required Minimum Distributions (RMDs)?",
    # Pension Risk Transfer
    "What is a pension risk transfer lift-out and why would a plan sponsor use it?",
    # Suitability red flag
    "What are the suitability red flags an agent should watch for when recommending an annuity?",
    # Edge case — no matching context
    "What is the process for filing a homeowners insurance claim?",
]


def run_demo(questions: list[str]) -> None:
    """
    Index synthetic documents then answer every question, printing results.

    Args:
        questions: List of question strings to run through the pipeline.
    """
    print("\n" + "=" * 70)
    print("  Athene RAG Pipeline — Practice Demo")
    print("=" * 70 + "\n")

    # ── Step 1: Load synthetic documents ─────────────────────────────────
    print("Loading synthetic insurance documents…")
    documents = get_synthetic_documents()
    print(f"  → {len(documents)} documents loaded.\n")

    # ── Step 2: Build FAISS index ─────────────────────────────────────────
    print("Building FAISS index (embedding all chunks)…")
    print("  BOTTLENECK: This step scales with corpus size.\n")
    vectorstore = build_index(documents)
    print()

    # ── Step 3: Answer each question ──────────────────────────────────────
    for i, question in enumerate(questions, start=1):
        print(f"{'─' * 70}")
        print(f"Q{i}: {question}")
        print()

        try:
            result = answer_question(question, vectorstore)
        except ValueError as e:
            print(f"  [INPUT ERROR] {e}\n")
            continue
        except RuntimeError as e:
            print(f"  [PIPELINE ERROR] {e}\n")
            continue

        print("ANSWER:")
        print(result["answer"])
        print()

        if result.get("sources"):
            print(f"Sources used: {', '.join(sorted(set(result['sources'])))}")

        print(
            f"Model: {result['model']}  |  "
            f"Est. tokens: {result.get('tokens', 'n/a')}"
        )
        print()

    print("=" * 70)
    print("Demo complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Allow passing a single question as a CLI arg for quick testing:
    #   python main.py "What perils are covered under Coverage A?"
    if len(sys.argv) > 1:
        custom_question = " ".join(sys.argv[1:])
        from rag_pipeline import build_index, answer_question
        docs = get_synthetic_documents()
        vs   = build_index(docs)
        result = answer_question(custom_question, vs)
        print("\nANSWER:\n", result["answer"])
        print("\nSources:", result.get("sources"))
    else:
        run_demo(DEMO_QUESTIONS)
