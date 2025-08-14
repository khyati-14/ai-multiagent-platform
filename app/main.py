from fastapi import FastAPI, File, UploadFile, Form
import shutil
import os
from app.services.rag_pipeline import ingest_file, query_vectorstore

app = FastAPI(title="AI Multi-Agent Platform Backend")

UPLOAD_DIR = "app/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return ingest_file(file_path)

@app.post("/query")
async def query_rag(query: str = Form(...)):
    return query_vectorstore(query)
