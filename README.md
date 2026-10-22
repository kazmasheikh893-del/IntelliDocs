# IntelliDocs: AI-Powered Document Intelligence Assistant Using RAG and Generative AI

IntelliDocs is an enterprise-grade Document Intelligence Assistant built with **Django 5.1**, **Retrieval-Augmented Generation (RAG)**, and the official **Google Gemini API** (`google-genai` SDK).

The system allows users to upload PDF or TXT documents, automatically parses and chunks text while strictly preserving page numbers, generates semantic vector embeddings, and answers user questions grounded exclusively in the uploaded context with transparent citations.

---

## 🌟 Key Features

1. **Multi-Format Document Ingestion**:
   - Handles `.pdf` (with page-level tracking via `pypdf`) and `.txt` files.
   - Robust file validation: MIME verification, file size limits (default 10 MB), and filename sanitization.
   - Rejects empty, corrupt, or password-protected files with clear user feedback.

2. **Page-Aware Chunking Engine**:
   - Custom recursive text chunking with configurable overlap (default 800 chars / 100 char overlap).
   - Preserves document name and PDF page numbers on every snippet so citations can always be verified.

3. **Vector Embeddings & Retrieval**:
   - Uses Google's official `text-embedding-004` (768 dimensions).
   - Lightweight, zero-dependency Django ORM + NumPy Cosine Similarity engine (no external vector database daemons required).
   - Clean `BaseVectorStore` interface allowing plug-and-play expansion to ChromaDB or pgvector.

4. **Grounded Answer Generation via Gemini**:
   - Utilizes `gemini-2.0-flash` (or `gemini-1.5-flash`).
   - Strict system grounding: answers strictly from retrieved passages.
   - Transparent citations formatted as `[Doc: filename, Page X]` alongside retrieved evidence cards.
   - **Deterministic Fallback**: If retrieved passages do not contain sufficient evidence, the system returns:  
     > *"The answer was not found in the provided documents."*
   - Strictly avoids hallucination, simulated citations, or mock AI outputs.

5. **Security & Configuration**:
   - `GEMINI_API_KEY` is loaded from `.env` and kept strictly on the backend.
   - `.env` is git-ignored; a safe `.env.example` is provided for reference.
   - Built-in verification command (`python manage.py verify_gemini`) to validate credentials against Google's API before deployment.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+ or 3.13 (64-bit)
- Google Gemini API Key (get one for free at [Google AI Studio](https://aistudio.google.com/))

### 2. Environment Setup

Activate the existing virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
Or create a fresh virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
pip install -r requirements.txt
```

### 3. Configure API Credentials

Copy the template file to `.env`:
```powershell
cp .env.example .env
```
Open `.env` and add your real Google Gemini API key:
```env
GEMINI_API_KEY=AIzaSyYourRealGeminiApiKeyHere
SECRET_KEY=your-custom-django-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 4. Verify Gemini API Connectivity

Run the built-in diagnostic command to test your credentials and model access:
```bash
python manage.py verify_gemini
```
Expected output upon successful configuration:
```text
==> Testing Google Gemini API Configuration...
[OK] Found GEMINI_API_KEY (AIzaSy...XYZ)
--> Testing Embedding Model: text-embedding-004 ...
[OK] Embedding SUCCESS (768 dimensions, latency: 450.2ms)
--> Testing Generation Model: gemini-2.0-flash ...
[OK] Generation SUCCESS (Response: 'READY', latency: 610.5ms)

[SUCCESS] All Gemini API checks passed successfully! Your RAG assistant is ready.
```

### 5. Apply Database Migrations & Start Server

```bash
python manage.py migrate
python manage.py runserver
```

Open your browser to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to access the application:
- **Repository Dashboard**: `http://127.0.0.1:8000/` (Upload and manage documents)
- **Q&A Assistant**: `http://127.0.0.1:8000/qa/` (Ask questions and view citations)
- **Django Admin**: `http://127.0.0.1:8000/admin/` (Inspect chunk records and embeddings)

---

## 🧪 Running the Test Suite

The project includes 26 unit and integration tests covering extraction, chunking, vector similarity search, grounding, and API endpoints:
```bash
python manage.py test documents
```

---

## 📂 Project Architecture

```
IntelliDocs/
├── manage.py
├── requirements.txt            # Pinned dependencies
├── .env.example                # Safe environment template
├── .gitignore                  # Prevents secret & media leaks
├── README.md                   # System documentation
├── intellidocs/                # Django core settings & routing
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── documents/                  # Document Intelligence & RAG App
│   ├── models.py               # Document & DocumentChunk schema
│   ├── forms.py                # Upload & file validation
│   ├── views.py                # Dashboard, upload & Q&A endpoints
│   ├── urls.py                 # Route definitions
│   ├── services/
│   │   ├── extractor.py        # PDF/TXT extraction with page metadata
│   │   ├── chunker.py          # Page-preserving text chunker
│   │   ├── embeddings.py       # Google GenAI text-embedding-004 client
│   │   ├── vector_store.py     # BaseVectorStore & ORMVectorStore (NumPy cosine)
│   │   ├── generator.py        # Gemini grounding, prompt assembly & QA
│   │   └── ingestion.py        # Ingestion pipeline coordinator
│   ├── management/commands/
│   │   └── verify_gemini.py    # Live API connectivity verification tool
│   └── tests/                  # 26 automated unit and integration tests
├── templates/                  # Responsive web interface
│   ├── base.html
│   └── documents/
│       ├── index.html          # Document repository & upload UI
│       └── chat.html           # Interactive grounded Q&A interface
└── static/
    ├── css/style.css           # Modern dark-slate UI styling
    └── js/app.js               # Async query submission & citation cards
```
