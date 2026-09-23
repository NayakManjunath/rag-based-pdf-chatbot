# RAG Based PDF Chat Bot

A Retrieval-Augmented Generation (RAG) based PDF chatbot that allows users to upload a PDF document and ask questions about its contents.

The application extracts text from the uploaded PDF, splits the document into chunks, generates embeddings using a Sentence Transformer model, stores the embeddings in a FAISS vector index, retrieves the most relevant document chunks for a user question, and sends the retrieved context to Gemini to generate a grounded answer.

If the requested information is not available in the retrieved document context, the application instructs the LLM to respond:

> I don't know from this document.

---

## Project Overview

The goal of this project is to build a simple, practical RAG application for question answering over user-provided PDF documents.

### Core Workflow

```text
User uploads PDF
       ↓
PDF text extraction
       ↓
Text chunking
       ↓
Embedding generation
       ↓
FAISS vector index
       ↓
User asks question
       ↓
Question embedding
       ↓
Similarity search
       ↓
Top-k relevant chunks
       ↓
Context construction
       ↓
Grounded Gemini prompt
       ↓
Answer
```

---

## Features

- Upload a PDF through a Streamlit interface
- Extract text from the uploaded PDF
- Split extracted text into fixed-size chunks
- Generate text embeddings using `all-MiniLM-L6-v2`
- Normalize generated embeddings
- Store embeddings in a FAISS `IndexFlatIP` index
- Generate an embedding for the user's question
- Retrieve the most relevant document chunks
- Construct a context from retrieved chunks
- Generate answers using Google Gemini
- Ground answers using retrieved PDF context
- Provide a fallback response when information is unavailable in the document
- Validate the application with automated component, retrieval, end-to-end, and RAG answer tests

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12.10 | Programming language |
| Streamlit | Web application interface |
| pypdf | PDF text extraction |
| Sentence Transformers | Text embedding generation |
| `all-MiniLM-L6-v2` | Embedding model |
| FAISS | Vector similarity search |
| Google Gemini | Answer generation |
| `google-genai` | Gemini API integration |
| python-dotenv | Environment variable management |
| pytest | Automated testing |

---

## Architecture

```text
                         ┌─────────────────────┐
                         │     Streamlit UI     │
                         └──────────┬──────────┘
                                    │
                              Upload PDF
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   PDF Processor     │
                         │                     │
                         │  Extract Text       │
                         │  Create Chunks      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Embeddings       │
                         │                     │
                         │ all-MiniLM-L6-v2    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    FAISS Index      │
                         │    IndexFlatIP      │
                         └──────────┬──────────┘
                                    │
                         User Question
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Question Embedding  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Retriever       │
                         │    Top-k Search     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   RAG Context       │
                         │    Construction     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Gemini LLM        │
                         │ Grounded Prompt     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              Final Answer
```

---

## Project Structure

```text
rag-based-pdf-chatbot/
│
├── data/
│   └── uploads/
│
├── src/
│   ├── pdf_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── llm.py
│
├── tests/
│   ├── test_embeddings.py
│   ├── test_end_to_end.py
│   ├── test_llm.py
│   ├── test_pdf_processor.py
│   ├── test_rag_answer_validation.py
│   ├── test_rag_pipeline.py
│   ├── test_retrieval_quality.py
│   ├── test_retriever.py
│   └── test_vector_store.py
│
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NayakManjunath/rag-based-pdf-chatbot.git
cd rag-based-pdf-chatbot
```

### 2. Create a Python 3.12 virtual environment

```bash
py -3.12 -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install project dependencies

```powershell
pip install -r requirements.txt
```

### 5. Install development/test dependencies

```powershell
pip install -r requirements-dev.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
```

The `.env` file is intentionally excluded from Git.

Use `.env.example` as the configuration reference.

---

## Running the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

If Streamlit's file watcher causes an optional dependency inspection issue, the application can be started with:

```powershell
streamlit run app.py --server.fileWatcherType none
```

The application will open in the browser.

---

## Using the Chatbot

### Step 1 — Upload a PDF

Use the PDF upload control in the Streamlit interface.

### Step 2 — Enter a question

Enter a question related to the uploaded document.

For example:

```text
What is iRAD?
```

### Step 3 — Ask the question

The application processes the PDF and retrieves relevant document chunks.

### Step 4 — Receive the answer

Gemini receives the retrieved document context and the question and generates the answer.

If the required information is not available in the document context, the configured fallback is:

```text
I don't know from this document.
```

---

## RAG Pipeline

### 1. PDF Processing

The uploaded PDF is saved under:

```text
data/uploads/
```

`pypdf` is used to extract text from the document.

### 2. Text Chunking

The extracted text is divided into fixed-size chunks.

The current implementation uses:

```text
chunk size = 800 characters
```

### 3. Embedding Generation

Each chunk is converted into an embedding using:

```text
all-MiniLM-L6-v2
```

Embeddings are normalized before being added to FAISS.

The resulting embeddings use:

```text
float32
```

### 4. FAISS Vector Store

The project uses:

```text
faiss.IndexFlatIP
```

Inner-product similarity is used because the embeddings are normalized.

### 5. Question Embedding

The user's question is converted into an embedding using the same embedding model.

### 6. Retrieval

FAISS searches for the most relevant chunks.

The current application retrieves:

```text
top-k = 3
```

chunks.

### 7. Context Construction

The retrieved chunks are combined into a single document context.

### 8. Grounded Generation

The context and question are passed to Gemini with a grounding instruction:

```text
Answer the question using only the provided document context.
```

The prompt also specifies the required fallback:

```text
I don't know from this document.
```

---

## Testing

The project uses `pytest` for automated testing.

Run the complete test suite:

```powershell
python -m pytest -v
```

### Final Test Result

```text
44 passed
```

| Testing Stage | Tests | Result |
|---|---:|---|
| Component Testing | 33 | PASS |
| Retrieval Testing | 5 | PASS |
| End-to-End Testing | 1 | PASS |
| RAG Answer Validation | 5 | PASS |
| **Total** | **44** | **44/44 PASS** |

---

## Testing Strategy

### Component Testing

Tests individual project components including:

- PDF processing
- Embedding generation
- FAISS vector store
- Question embedding
- Retrieval
- RAG context construction
- Grounded prompt construction

### Retrieval Testing

Tests whether semantically relevant chunks are retrieved for representative questions.

The tests also verify:

- top-k behavior
- deterministic retrieval
- relevant chunk selection

### End-to-End Testing

Tests the complete local RAG pipeline:

```text
PDF processing
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS
    ↓
Question embedding
    ↓
Retrieval
    ↓
Context construction
    ↓
Grounded prompt
```

The automated E2E test does not make a live Gemini API request.

### RAG Answer Validation

Validates that the grounded prompt:

- contains the document context
- contains the user question
- contains the grounding instruction
- contains the required fallback instruction
- preserves the supplied document context

---

## Manual Application Validation

The application was also manually validated using a real uploaded PDF.

The validation confirmed that:

1. A real PDF could be uploaded.
2. PDF text could be extracted.
3. Document chunks could be created.
4. Embeddings could be generated.
5. FAISS could retrieve relevant chunks.
6. Gemini could generate a document-grounded answer.
7. An out-of-document question produced:

```text
I don't know from this document.
```

---

## Configuration Details

The embedding model is configured for local loading:

```text
all-MiniLM-L6-v2
```

The project uses local model loading to avoid unnecessary Hugging Face network metadata checks during application execution.

The environment variables:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

are configured in the embedding module.

---

## Limitations

The current implementation intentionally keeps the project focused on a simple text-based PDF RAG workflow.

The current scope does not include:

- OCR for scanned/image-only PDFs
- Multiple simultaneous PDF collections
- Persistent document databases
- Chat conversation memory
- Hybrid BM25 + vector retrieval
- Cross-encoder reranking
- Agentic workflows
- Multi-modal document understanding
- Docker deployment
- Kubernetes deployment
- Cloud deployment
- CI/CD pipeline

These can be considered future extensions rather than current project capabilities.

---

## Future Improvements

Possible future extensions include:

1. OCR support for scanned PDFs
2. Multiple PDF management
3. Persistent vector storage
4. Conversation memory
5. Hybrid keyword + vector retrieval
6. Cross-encoder reranking
7. Improved chunking strategies
8. Source/page-level citations
9. Authentication and user management
10. Cloud deployment
11. Docker containerization
12. CI/CD automation

These are optional future improvements and are not part of the current implemented scope.

---

## Git Development Milestones

The project was developed incrementally with Git milestones.

```text
fba842f  feat(module-1): complete foundation & project setup
25db4c8  feat(module-2): implement PDF processing pipeline
3e68a9d  feat(module-4): implement retrieval and RAG context
bf78ff6  feat(module-5): integrate Gemini answer generation
02a5723  feat(module-6): complete PDF chatbot application
fad8c3b  test(module-7.1): add component test suite
f54a44b  test(module-7.2): add retrieval quality tests
9a4aa62  test(module-7.3): add end-to-end RAG test
03b8aa5  test(module-7.4): add RAG answer validation
```

The final repository state was verified with:

```text
HEAD -> main
origin/main
working tree clean
```

---

## Project Status

```text
Modules Completed: 7/7

Module 1  Foundation & Project Setup       ✅
Module 2  PDF Processing                   ✅
Module 3  Embedding & Vector Store         ✅
Module 4  Retrieval & RAG Pipeline         ✅
Module 5  LLM Answer Generation            ✅
Module 6  PDF Chatbot Application           ✅
Module 7  Testing & Validation              ✅

Automated Tests: 44/44 PASS
```

---

## Key Learning Outcomes

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation
- PDF document processing
- Text chunking
- Sentence Transformer embeddings
- Vector similarity search
- FAISS
- Semantic retrieval
- Context construction
- Prompt grounding
- Gemini API integration
- Streamlit application development
- Environment configuration
- Automated testing with pytest
- Component testing
- Retrieval testing
- End-to-end testing
- Git-based incremental development

---

## Author

**Manjunath Naik**

GitHub:

https://github.com/NayakManjunath
