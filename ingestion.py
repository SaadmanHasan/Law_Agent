import csv
from pathlib import Path
from typing import List

import fitz  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATA_DIR = Path("data")
DOCS_DIR = Path("uploads/docs")
VECTOR_DIR = Path("vectorstore")

CHAT_CSV = DATA_DIR / "chat_history.csv"

def build_chat_documents() -> List[Document]:
    docs: List[Document] = []
    if not CHAT_CSV.exists():
        return docs

    with CHAT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            msg = row.get("Message", "").strip()
            if not msg:
                continue
            date = row.get("Date", "")
            time = row.get("Time", "")
            sender = row.get("Sender", "")
            src = row.get("Source", "chat_history.csv")
            content = f"{sender} at {time} on {date}: {msg}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": src,
                        "row_id": idx,
                        "date": date,
                        "time": time,
                        "sender": sender,
                    },
                )
            )
    return docs

def build_pdf_documents() -> List[Document]:
    docs: List[Document] = []
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in DOCS_DIR.glob("*.pdf"):
        doc = fitz.open(pdf_path)
        filename = pdf_path.stem  
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            content = f"[Document: {filename}]\n{text}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": pdf_path.name,
                        "page": i,
                        "filename": filename,
                    },
                )
            )
    return docs

def rebuild_vectorstore():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    VECTOR_DIR.mkdir(exist_ok=True, parents=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    docs = build_chat_documents() + build_pdf_documents()

    import shutil
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="case_docs",
        persist_directory=str(VECTOR_DIR),
    )
    print(f"Indexed {len(docs)} documents.")

if __name__ == "__main__":
    rebuild_vectorstore()
