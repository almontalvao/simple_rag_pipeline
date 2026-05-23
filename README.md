# RAG Pipeline — Practice Project

Simple Production-grade RAG system. Includes embeddings,
robust retrieval, grounded prompting, edge case handling, and bottleneck awareness.

---

## Setup

### 1. Create a virtual environment

**Windows (Command Prompt):**
```
python -m venv venv
```

**Windows (PowerShell):**
```
python -m venv venv
```

### 2. Activate the virtual environment

**Windows (Command Prompt):**
```
venv\Scripts\activate
```

**Windows (PowerShell):**
```
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run this first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install dependencies

```
pip install -r requirements.txt
```

> First run will download the `all-MiniLM-L6-v2` sentence-transformers model (~90 MB).
> Subsequent runs use the cached version and start immediately.

---

## Running the pipeline

### Demo mode (no API key required)

```
python main.py
```

Uses local sentence-transformers for embeddings. The LLM generation step returns
a structured mock response showing the retrieved context and the exact prompt that
would be sent to GPT-4o.

### Production mode (OpenAI API key)

**Windows (Command Prompt):**
```
set OPENAI_API_KEY=sk-...
python main.py
```

**Windows (PowerShell):**
```
$env:OPENAI_API_KEY="sk-..."
python main.py
```

Switches automatically to `text-embedding-3-small` for embeddings and `gpt-4o`
for generation. The full 8-question demo run costs under $0.10.

### Ask a single question

```
python main.py "How does the GLWB Benefit Base differ from the account value?"
```

---

## Project structure

```
code_prep/
├── rag_pipeline.py      # Complete RAG pipeline (Section 6 of the prep guide)
├── synthetic_data.py    # Synthetic Athene annuity and retirement documents
├── main.py              # Demo runner — builds index and answers 8 questions
└── requirements.txt     # All dependencies
```

---

## Deactivating the virtual environment

When done:
```
deactivate
```
