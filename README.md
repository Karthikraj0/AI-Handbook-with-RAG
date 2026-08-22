# 📖 AI Handbook & Knowledge Base (RAG Assistant)

A production-ready Retrieval-Augmented Generation (RAG) assistant designed to provide accurate, grounded answers to employee inquiries regarding company policies, benefits, travel, leave guidelines, and operational procedures.

Built with **Streamlit**, **PyMuPDF**, **ChromaDB**, and **Ollama** (`gpt-oss` / `nomic-embed-text`).

---

## ✨ Features

- **Zero Hallucination Policy**: Strictly grounds answers in company policy PDF documents.
- **Source Transparency**: Every response provides citations pointing to the exact source PDF and page number.
- **Real-Time Token Streaming**: Low-latency token generation powered by Ollama and Streamlit `st.write_stream`.
- **Vintage Neo-Brutalist UI**: Premium typography (`Space Grotesk`, `Playfair Display`, `Space Mono`) and dynamic chat experience.
- **Modular Pipeline**: Clean separation of document loading, text chunking, embedding generation, vector storage, retrieval, and LLM generation.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Ingestion_Pipeline ["Ingestion (Offline)"]
        A["Company Policy PDFs (documents/*.pdf)"] --> B["PyMuPDF Loader (rag/loader.py)"]
        B --> C["Sliding-Window Chunker (rag/chunker.py)"]
        C --> D["Embedding Generator (rag/embeddings.py)"]
        D --> E["ChromaDB Collection (rag/vectorstore.py)"]
    end

    subgraph Query_Pipeline ["Query Execution (Runtime)"]
        Q["Employee Question (UI / app.py)"] --> Pipeline["rag/pipeline.py"]
        Pipeline --> Rewriter["rag/query_rewriter.py (qwen3:1.7b)"]
        Rewriter --> Retrieve["rag/retrieve.py"]
        Retrieve --> E
        E --> |Top-5 Chunks (Filtered <= 0.47)| Pipeline
        Pipeline --> Generator["rag/generator.py (Ollama llama3.2:3b)"]
        Generator --> |Token Stream| Streamlit["Streamlit UI (app.py)"]
        Streamlit --> |Real-time Answer + Sources| User
    end
```

---

## 📁 Repository Structure

```
├── app.py                      # Streamlit frontend with neo-brutalist styling & streaming
├── DOCUMENTATION.md            # Detailed technical documentation & architectural manual
├── HOW_TO_RUN.txt              # Step-by-step local setup guide
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit server and theme configurations
├── documents/                  # Policy documents (PDF format)
│   ├── attendance_policy.pdf
│   ├── employee_benefits.pdf
│   ├── employee_handbook.pdf
│   ├── it_security_policy.pdf
│   ├── leave_policy.pdf
│   ├── reimbursement_policy.pdf
│   ├── travel_policy.pdf
│   └── work_from_home_policy.pdf
└── rag/                        # Modular RAG backend package
    ├── __init__.py             # Package marker
    ├── loader.py               # PyMuPDF text extraction
    ├── chunker.py              # Sliding-window document chunking
    ├── embeddings.py           # Ollama embeddings (nomic-embed-text)
    ├── vectorstore.py          # ChromaDB persistence & collection manager
    ├── ingest.py               # Document ingestion pipeline
    ├── retrieve.py             # Semantic similarity retrieval
    ├── generator.py            # Prompt builder & streaming LLM generator
    └── pipeline.py             # End-to-end query pipeline
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.9+ installed
- [Ollama](https://ollama.com/) installed and running locally with the required models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen3:1.7b
  ollama pull llama3.2:3b
  ```

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd "AI Handbook with RAG"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ingest Documents (Optional if data already exists)
```bash
python -m rag.ingest
```

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📜 License
This project is licensed under the MIT License.
