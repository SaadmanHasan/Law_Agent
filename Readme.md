# Case Research Agent - WhatsApp OCR & PDF Analysis

A Flask-based intelligent case research assistant that performs OCR on WhatsApp chat screenshots and searches through PDFs using semantic search and AI-powered question answering.

## Features

- 📸 **OCR Processing**: Extract text from WhatsApp chat screenshots using EasyOCR
- 📄 **PDF Analysis**: Upload and index PDF documents (invoices, purchase orders, etc.)
- 🔍 **Semantic Search**: Find relevant information across documents using embeddings
- 🤖 **AI Q&A**: Ask natural language questions and get answers based on indexed documents
- 💬 **Multi-source**: Search both WhatsApp chats and PDFs simultaneously
- 📋 **Evidence Citations**: Every answer includes source documents with page/row numbers

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- OpenRouter API key (for LLM access)

## Installation & Setup

### Step 1: Create a Virtual Environment

On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

On Windows (Command Prompt):
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

On macOS/Linux:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root directory:

```
OPENROUTER_API_KEY=your_api_key_here
```

Get your API key from: https://openrouter.ai/

### Step 4: Run the Application

```powershell
python app.py
```

The application will start on `http://localhost:3000`

## Usage

### 1. Upload WhatsApp Chat Screenshots

- Go to the **"Upload case PDFs"** section (or create an image upload section)
- Upload one or more WhatsApp chat screenshot images
- The system will perform OCR and extract chat messages
- Chat history is saved to `data/chat_history.csv`

### 2. Upload PDF Documents

- Click **"Browse..."** to select PDF files
- Click **"Upload & Reindex"** to add documents to the search index
- PDFs are stored in `uploads/docs/`

### 3. Ask Questions

- Navigate to the **"Ask a question"** section
- Enter your question (e.g., "Did I get MU7339?" or "What is the total on the latest invoice?")
- Click **"Ask"**
- View the AI-generated answer and supporting evidence below

## Project Structure

```
WA_ocr/
├── app.py                 # Flask application & routes
├── qa_chain.py            # LLM agent & semantic search logic
├── ingestion.py           # PDF & CSV processing, vector indexing
├── text_recog.py          # OCR processing for images
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── data/
│   └── chat_history.csv   # Extracted chat messages
├── uploads/
│   ├── docs/              # Uploaded PDF files
│   └── images/            # Uploaded WhatsApp screenshots
├── vectorstore/           # Chroma vector database
│   └── chroma.sqlite3     # Embedded index
├── templates/
│   ├── index.html         # Image upload page
│   └── qa.html            # Q&A interface
└── static/
    └── style.css          # Styling
```

## How It Works

### Data Flow

1. **OCR Phase**: WhatsApp screenshots → Text extraction → CSV storage
2. **Ingestion Phase**: PDFs + Chat CSV → Text chunks → Embeddings → Vector DB
3. **Query Phase**: Question → Semantic search → Context assembly → LLM call → Answer

### Key Technologies

- **LangChain**: LLM orchestration and tool management
- **Chroma**: Vector database for semantic search
- **HuggingFace Embeddings**: Text-to-vector conversion (sentence-transformers/all-MiniLM-L6-v2)
- **EasyOCR**: Text recognition from images
- **OpenRouter**: LLM API (Mistral 7B)
- **PyMuPDF**: PDF text extraction
- **Flask**: Web framework
