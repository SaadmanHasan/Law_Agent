import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

VECTOR_DIR = Path("vectorstore")

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},          
        encode_kwargs={"normalize_embeddings": False},
    )

def get_vectorstore() -> Chroma:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    return Chroma(
        collection_name="case_docs",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )


def _search_case_knowledge(query: str) -> str:

    vs = get_vectorstore()
    docs: List[Document] = vs.similarity_search(query, k=8)

    if not docs:
        return "No relevant documents were found for this query."

    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        row_id = meta.get("row_id")
        page = meta.get("page")
        if row_id is not None:
            src = f"{src} (row {row_id})"
        elif page is not None:
            src = f"{src} (page {page})"
        snippet = (d.page_content or "").replace("\n", " ")
        lines.append(f"[{i}] {snippet}  (Source: {src})")

    return "\n".join(lines)


class RetrievalInput(BaseModel):
    query: str = Field(
        description="Natural language query to search within the case documents and chat history."
    )


@tool("case_retrieval", args_schema=RetrievalInput)
def retrieval_tool(query: str) -> str:
    """
    Use this to search the WhatsApp chat CSV and uploaded PDFs for information related to the user's question.
    Returns numbered snippets with their sources.
    """
    return _search_case_knowledge(query)


SYSTEM_PROMPT = """You are a case research assistant.
You can use the `case_retrieval` tool to search WhatsApp chat history and case PDFs (purchase orders, invoices, etc.).
When answering a question:
- First think whether you need to call the tool; if the answer requires case-specific facts (dates, amounts, POs, etc.), you must call it.
- Read the returned snippets carefully and base your answer only on that information.
- Always cite the snippets you used by their bracket number and source, for example: [1] (Source: chat_history.csv, row 15).
- If the information is not present in the snippets, say that it is not available in the case documents.
Respond concisely and factually.
"""

model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",  
        "X-Title": "WhatsApp Case Research Assistant",  
    }
)

model_with_tools = model.bind_tools([retrieval_tool])

def answer_question(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Run the agent on the user's question and also return explicit evidence.
    chat_history: optional list of dicts like {"role": "user"|"assistant", "content": "..."}
    """

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    if chat_history:
        for msg in chat_history:
            role = (msg.get("role") or "").lower()
            content = msg.get("content") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    

    messages.append(HumanMessage(content=question))
    
    vs = get_vectorstore()
    docs: List[Document] = vs.similarity_search(question, k=8)
    
    context = ""
    if docs:
        lines = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            src = meta.get("source", "unknown")
            row_id = meta.get("row_id")
            page = meta.get("page")
            if row_id is not None:
                src = f"{src} (row {row_id})"
            elif page is not None:
                src = f"{src} (page {page})"
            snippet = (d.page_content or "").replace("\n", " ")
            lines.append(f"[{i}] {snippet}  (Source: {src})")
        context = "\n".join(lines)
    
    if context:
        messages[-1] = HumanMessage(content=f"Based on this information:\n{context}\n\nAnswer this question: {question}")
    
    response = model_with_tools.invoke(messages)
    answer = response.content if hasattr(response, 'content') else str(response)

    evidence = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        row_id = meta.get("row_id")
        page = meta.get("page")
        if row_id is not None:
            src = f"{src} (row {row_id})"
        elif page is not None:
            src = f"{src} (page {page})"
        snippet = (d.page_content or "").replace("\n", " ")
        evidence.append({"source": src, "snippet": snippet})

    return {"answer": answer, "sources": evidence}

# if __name__ == "__main__":
#     resp = answer_question("What is the total on the latest invoice?")
#     print(resp["answer"])
#     for ev in resp["sources"]:
#         print("-", ev["source"])
