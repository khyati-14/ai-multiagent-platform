import os
import mimetypes
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredHTMLLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import VECTOR_DB_PATH

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_loader(file_path: str):
    """Return appropriate document loader based on file type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if mime_type == "application/pdf" or ext == ".pdf":
        return PyPDFLoader(file_path)

    elif mime_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ] or ext in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(file_path)

    elif mime_type == "text/csv" or ext == ".csv":
        return CSVLoader(file_path)

    elif mime_type == "text/html" or ext == ".html":
        return UnstructuredHTMLLoader(file_path)

    elif mime_type and mime_type.startswith("text") or ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8")

    else:
        return TextLoader(file_path, encoding="utf-8")  # Fallback

from pdf2image import convert_from_path
import pytesseract

def ingest_file(file_path: str):
    """Load, split, embed and store document, with OCR fallback."""
    try:
        # Step 1: Try normal PDF/Text loading
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()

        # Step 2: Fallback to OCR if no text
        if not documents or all(len(doc.page_content.strip()) == 0 for doc in documents):
            print("⚠ No text found — running OCR...")
            documents = []
            images = convert_from_path(file_path)
            for img in images:
                text = pytesseract.image_to_string(img)
                if text.strip():
                    from langchain.schema import Document
                    documents.append(Document(page_content=text))

            if not documents:
                return {"status": "error", "message": "OCR failed — no text could be extracted."}

        # Step 3: Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        if not docs:
            return {"status": "error", "message": "No valid chunks created from document."}

        # Step 4: Store in FAISS
        if os.path.exists(VECTOR_DB_PATH):
            vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(docs)
        else:
            vectorstore = FAISS.from_documents(docs, embeddings)

        vectorstore.save_local(VECTOR_DB_PATH)
        return {"status": "success", "chunks": len(docs)}

    except Exception as e:
        return {"status": "error", "message": str(e)}



def query_vectorstore(query: str):
    """Query the vectorstore."""
    if not os.path.exists(VECTOR_DB_PATH):
        return {"error": "Vectorstore not found. Please ingest a document first."}

    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=3)

    return {"matches": [r.page_content for r in results]}
