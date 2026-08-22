# AI Handbook RAG System — Technical Documentation & Architecture Manual

This document provides a comprehensive, file-by-file technical breakdown and end-to-end architectural manual for the **AI Handbook with RAG** application.

---

## 1. Executive Overview

The **AI Handbook RAG System** is a Retrieval-Augmented Generation (RAG) assistant designed to provide accurate, grounded answers to employee inquiries regarding company policies, benefits, travel, leave guidelines, and operational procedures.

### Core Objectives:
1. **Zero Hallucination Policy**: The assistant answers strictly using the extracted organizational policy documents. If the requested information is absent, it responds with:
   > *"I couldn't find that information in the company policies."*
2. **Transparent Grounding**: Answers are accompanied by real document citations (file name and page number).
3. **High-Performance UX**: Low-latency responses powered by Ollama memory caching, ChromaDB in-memory singletons, real-time token streaming (`st.write_stream`), and a custom 3-dot typing indicator animation.
4. **Vintage Neo-Brutalist UI**: An interface built with Streamlit and styled using custom CSS (Space Grotesk, Space Mono, Playfair Display typography with a retro cream/black palette).

---

## 2. High-Level Architecture

```mermaid
graph TD
    subgraph Ingestion_Phase ["Offline Ingestion Pipeline"]
        PDFs["PDF Documents (documents/*.pdf)"] --> Loader["rag/loader.py (PyMuPDF)"]
        Loader --> Chunker["rag/chunker.py (Sliding Window)"]
        Chunker --> Embedder["rag/embeddings.py (nomic-embed-text)"]
        Embedder --> VectorDB["rag/vectorstore.py (ChromaDB)"]
    end

    subgraph Runtime_Phase ["Runtime RAG Execution"]
        User["User Question (Streamlit UI / app.py)"] --> Pipeline["rag/pipeline.py (ask_question_stream)"]
        Pipeline --> Rewriter["rag/query_rewriter.py (Ollama Reformulation)"]
        Rewriter --> |Normalized Query| Retrieve["rag/retrieve.py"]
        Retrieve --> VectorDB
        VectorDB --> |Top-5 Context & Sources (Filtered <= 0.47)| Pipeline
        Pipeline --> Generator["rag/generator.py (Ollama gpt-oss:latest)"]
        Generator --> |Token Stream| Streamlit["Streamlit UI (app.py)"]
        Streamlit --> |Real-time Tokens + Source Citations| User
    end
```

---

## 3. Directory & File Inventory

```
AI Handbook with RAG/
│
├── app.py                      # Streamlit frontend with custom CSS, chat history & streaming
├── check.py                    # Standalone CLI testing script for the backend pipeline
├── test_50_benchmark.py        # Comprehensive 154-question accuracy & confusion matrix benchmark
├── test_distance_gap.py        # Distance threshold gap evaluation benchmark
├── DOCUMENTATION.md            # Complete technical documentation (this file)
├── HOW_TO_RUN.txt              # Quickstart guide for running the application
├── requirements.txt            # Python dependencies
│
├── documents/                  # Source policy documents (PDF format)
│   ├── attendance_policy.pdf
│   ├── employee_benefits.pdf
│   ├── employee_handbook.pdf
│   ├── it_security_policy.pdf
│   ├── leave_policy.pdf
│   ├── reimbursement_policy.pdf
│   ├── travel_policy.pdf
│   └── work_from_home_policy.pdf
│
├── data/                       # Persistent vector database storage
│   └── chroma/                 # ChromaDB parquet/sqlite vector indices
│
└── rag/                        # Modular RAG backend package
    ├── __init__.py             # Package marker
    ├── loader.py               # PDF document text extractor
    ├── chunker.py              # Text sliding-window chunker
    ├── embeddings.py           # Vector embedding generator (Ollama)
    ├── vectorstore.py          # ChromaDB persistence & collection manager
    ├── ingest.py               # Ingestion pipeline orchestration
    ├── retrieve.py             # Semantic similarity retrieval engine
    ├── generator.py            # LLM prompt builder & streaming generator
    └── pipeline.py             # Main entry point combining retrieval & generation
```

---

## 4. Deep-Dive: File-by-File Breakdown

---

### `app.py` — Streamlit Frontend & Design Engine
- **Role**: Serves as the web application interface, handling user interaction, layout management, custom CSS styling, chat history state, 3-dot typing animations, and live response streaming.
- **Key Modules & Dependencies**:
  - `streamlit`: Web application framework.
  - `rag.pipeline.ask_question_stream`: Asynchronous token streaming interface to the RAG backend.
- **Internal Mechanisms**:
  1. **Page Config & Styling**: Configures wide layout and injects comprehensive CSS featuring custom web fonts (`Playfair Display`, `Space Grotesk`, `Space Mono`), custom scrollbars, neo-brutalist buttons, and card shadows.
  2. **Session State Management**:
     - `st.session_state.messages`: Stores conversation history (`role`, `content`, `sources`).
     - `st.session_state.history_queries`: Stores unique past questions shown in the sidebar.
     - `st.session_state.pending_prompt`: Tracks user prompt to be processed in the current render cycle.
  3. **Sidebar Components**:
     - "＋ New Chat" button to reset the session.
     - "About" card explaining the assistant's purpose.
     - "Recent Questions" list (last 5 queries).
     - System Status telemetry (Retrieval: ONLINE, LLM: OLLAMA, Mode: RAG).
  4. **Welcome Screen & Suggested Questions**:
     - Displays vintage welcome hero card when no messages exist.
     - Provides 4 preset query buttons ("Remote work policy", "Reimbursement policy", "Request time off", "Employee benefits").
  5. **3-Dot Loading Indicator (`.typing-indicator`)**:
     - When a question is submitted, an empty placeholder renders 3 animated bouncing dots inside the assistant bubble while ChromaDB retrieves context and Ollama initializes the first token.
  6. **Token Streaming (`st.write_stream`)**:
     - Consumes chunks from `ask_question_stream()` and updates the UI in real time.
  7. **Source Citations Card (`render_sources_card`)**:
     - Extracts source metadata (`source`, `page`), deduplicates identical citations, and renders them in a formatted sources card below the response text.
     - Automatically suppresses source display if the model returned the fallback "not found" response.

---

### `rag/pipeline.py` — End-to-End Pipeline Orchestrator
- **Role**: Glues the retrieval module and the generation module together into simple, callable functions.
- **Functions**:
  - `ask_question(query)`:
    - Calls `retrieve_documents(query, n_results=5)` to fetch top-5 relevant policy chunks.
    - Joins chunk texts into a unified `context` string.
    - Calls `generate_answer(query, context)` to get the full blocking response.
    - Returns `{"answer": str, "sources": list, "reformulated_query": str}`.
  - `ask_question_stream(query)`:
    - Calls `reformulate_query(query)` to expand abbreviations, resolve typos, and clarify intent.
    - Calls `retrieve_documents(query, n_results=5, reformulated_query=reformulated_query)`.
    - If no relevant documents exist (distance > 0.47), returns a fallback generator yielding the "not found" string.
    - Otherwise, calls `generate_answer_stream(query, context)` using the original user query and returns `{"stream": generator, "sources": list, "reformulated_query": str}`.

---

### `rag/query_rewriter.py` — LLM-Based Query Reformulation
- **Role**: Normalizes user queries prior to embedding and ChromaDB retrieval. Expands workplace acronyms (`wfh` $\rightarrow$ `work from home`, `pto` $\rightarrow$ `paid time off`, `lop` $\rightarrow$ `loss of pay`, etc.), corrects spelling/grammar mistakes, and converts short conversational phrases into searchable policy terms.
- **Key Settings**:
  - `REWRITER_MODEL = "gpt-oss:latest"`
  - `temperature = 0.0` (fully deterministic query reformulation)
  - `keep_alive = "1h"`
- **Prompt Guardrails**:
  - Strictly forbidden from answering the question or hallucinating policy facts.
  - Returns ONLY the concise search query string.
- **Resilience / Safe Fallback**:
  - If the LLM call times out, encounters an exception, or returns an empty/unusable string, it automatically falls back to the original user query without failing the pipeline.
- **Functions**:
  - `reformulate_query(query: str) -> str`: Transforms raw user query into semantic retrieval-ready search query.

---

### `rag/generator.py` — LLM Prompting & Response Generation
- **Role**: Manages model interaction with Ollama (`gpt-oss:latest`) and enforces strict anti-hallucination prompt constraints.
- **Key Settings**:
  - `LLM_MODEL = "gpt-oss:latest"`
  - `temperature = 0.2` (enforces strict deterministic, factual, low-hallucination outputs)
  - `keep_alive = "1h"` (ensures model stays resident in RAM/VRAM across user queries).
- **Functions**:
  - `_build_prompt(query, context)`:
    - Constructs the prompt with explicit guardrails:
      - *"Answer using ONLY the information provided in the policy context."*
      - *"If the answer cannot be found, say: 'I couldn't find that information in the company policies.'"*
      - *"Do not make up information or use outside knowledge."*
  - `generate_answer(query, context)`: Blocking chat completion returning full text string with `options={"temperature": 0.2}`.
  - `generate_answer_stream(query, context)`: Generator yielding individual token chunks via `ollama.chat(..., stream=True, options={"temperature": 0.2})` as they arrive.

---

### `rag/retrieve.py` — Semantic Search & Document Retrieval
- **Role**: Translates user search queries into embeddings and queries the vector database.
- **Functions**:
  - `retrieve_documents(query, n_results=5)`:
    1. Obtains the persistent ChromaDB collection via `get_collection()`.
    2. Generates an embedding vector for the incoming user query using `create_embedding(query)`.
    3. Queries ChromaDB using vector similarity (`n_results=5`).
    4. Returns a results dictionary containing `documents` (text chunks), `metadatas` (document name and page number), and `distances`.

---

### `rag/embeddings.py` — Vector Embedding Generation
- **Role**: Communicates with Ollama's embedding engine to generate high-dimensional vector representations of text.
- **Key Settings**:
  - `EMBEDDING_MODEL = "nomic-embed-text"`
  - `keep_alive = "1h"` (prevents unloading when switching between embedding and generation).
- **Functions**:
  - `create_embedding(text)`: Generates and returns a single embedding vector (list of floats).
  - `create_embeddings(texts)`: Iterates over a list of texts and returns a list of embedding vectors.

---

### `rag/vectorstore.py` — Vector Database Storage Manager
- **Role**: Manages the local persistent ChromaDB vector store.
- **Key Settings**:
  - `CHROMA_PATH = "data/chroma"`
  - `COLLECTION_NAME = "company_policies"`
- **Internal Optimization**:
  - Uses a module-level cached singleton (`_collection`) so `chromadb.PersistentClient` and the collection are initialized once in memory, eliminating redundant disk I/O per query.
- **Functions**:
  - `get_collection()`: Returns the cached collection instance, initializing it on first call.
  - `add_chunks(collection, chunks, embeddings)`: Inserts chunk IDs (`chunk_0`, `chunk_1`, ...), chunk text bodies, metadata dictionaries (`{"source": filename, "page": page_num}`), and embedding vectors into ChromaDB.

---

### `rag/chunker.py` — Text Segmentation Engine
- **Role**: Splits long document pages into smaller, overlapping chunks suitable for semantic embedding and LLM context windows.
- **Algorithm**:
  - **Chunk Size**: `150` words
  - **Overlap**: `30` words (provides semantic continuity across boundaries)
  - **Sliding Window**: Advances by `chunk_size - overlap` (120 words) per step.
- **Functions**:
  - `chunk_documents(documents, chunk_size=150, overlap=30)`: Converts a list of raw document page objects into segmented chunk objects preserving `source` and `page` metadata.

---

### `rag/loader.py` — PDF Ingestion & Text Extraction
- **Role**: Extracts plain text and page metadata from PDF policy files using PyMuPDF.
- **Functions**:
  - `load_pdfs(folder_path)`:
    - Scans the folder for all `*.pdf` files.
    - Opens each PDF using `pymupdf.open(pdf_file)`.
    - Iterates over pages, extracting text via `page.get_text("text")`.
    - Collects non-empty pages as dictionaries: `{"text": text, "source": pdf_file.name, "page": page_number}`.

---

### `rag/ingest.py` — Ingestion Pipeline Script
- **Role**: Offline utility script to populate ChromaDB from the `documents/` folder.
- **Execution Flow**:
  1. `load_pdfs("documents")` $\rightarrow$ Extracts all PDF pages.
  2. `chunk_documents(...)` $\rightarrow$ Splits text into 150-word chunks with 30-word overlap.
  3. `create_embeddings(...)` $\rightarrow$ Computes vector embeddings via `nomic-embed-text`.
  4. `get_collection()` & `add_chunks(...)` $\rightarrow$ Stores chunks and vectors in `data/chroma/`.
- **Usage**: Run when adding, updating, or replacing policy PDFs:
  ```powershell
  python -m rag.ingest
  ```

---

### `check.py` — Direct Pipeline CLI Verification
- **Role**: Standalone CLI test script used to verify that the retrieval and generation pipeline functions correctly without starting the Streamlit web server.
- **Code**:
  ```python
  from rag.pipeline import ask_question

  query = "How many annual leave days do employees get?"
  result = ask_question(query)

  print("ANSWER:")
  print(result["answer"])
  print("\nSOURCES:")
  for source in result["sources"]:
      print(source)
  ```

---

## 5. Lifecycle Step-by-Step Data Flow

```
1. USER ENTERS QUESTION (e.g., "How many annual leave days do employees get?")
   │
   ▼
2. STREAMLIT FRONTEND (app.py)
   ├─ Appends User message to st.session_state.messages
   ├─ Displays Assistant bubble with 3-dot loading animation (.typing-indicator)
   └─ Calls rag.pipeline.ask_question_stream(query)
   │
   ▼
3. SEMANTIC RETRIEVAL (rag/retrieve.py)
   ├─ Calls rag.embeddings.create_embedding(query) via Ollama (nomic-embed-text)
   ├─ Queries ChromaDB collection for top-5 nearest neighbor chunks
   └─ Extracts retrieved texts + metadata (e.g., leave_policy.pdf, Page 1)
   │
   ▼
4. PROMPT ASSEMBLY & STREAMING (rag/generator.py)
   ├─ Fills POLICY CONTEXT and EMPLOYEE QUESTION into strict prompt template
   ├─ Calls ollama.chat(model="gpt-oss:latest", stream=True, keep_alive="1h")
   └─ Yields tokens back through generator
   │
   ▼
5. LIVE STREAMING TO FRONTEND (app.py)
   ├─ st.write_stream(stream) replaces 3 dots with live word-by-word text
   ├─ Validates whether answer found information in policies
   ├─ Deduplicates & renders real source cards below the text
   └─ Appends assistant response to st.session_state.messages
```

---

## 6. Performance & Quality Guarantees

| Metric / Feature | Implementation Mechanism | Benefit |
| :--- | :--- | :--- |
| **Response Latency** | `st.write_stream` with Ollama streaming | User sees tokens immediately (< 1s time-to-first-token) |
| **Memory Resident Models** | `keep_alive="1h"` in Ollama embed & chat calls | Eliminates 3–8s model unloading/reloading delay |
| **Vector DB Speed** | Singleton caching in `rag/vectorstore.py` | Eliminates repeated ChromaDB disk index initialization |
| **Hallucination Protection** | Strict negative constraint prompt engineering | Out-of-domain queries return fallback with no fake citations |
| **Visual Aesthetics** | Vintage Cream/Black Neo-Brutalist CSS & Typing Dots | Responsive, modern, and polished user experience |

---

## 7. How to Operate & Maintain

### 1. Starting the Application
```powershell
streamlit run app.py
```

### 2. Ingesting New Policy PDFs
1. Place the new `.pdf` files into the `documents/` folder.
2. Run the ingestion command:
   ```powershell
   python -m rag.ingest
   ```
3. Restart the Streamlit app.

### 3. Testing Backend via Terminal
```powershell
python check.py
```
