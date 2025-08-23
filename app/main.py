import os
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.services.rag_pipeline import ingest_document, query_rag
from app.agents.agent_manager import run_multiagent_query
# from app.agents.langgraph_pipeline import run_langgraph_query
from app.config import VECTOR_DB_PATH
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Sentry for error monitoring
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    environment=os.getenv("ENVIRONMENT", "development")
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Multi-Agent Platform")
    yield
    logger.info("Shutting down AI Multi-Agent Platform")

app = FastAPI(
    title="AI Multi-Agent Platform",
    description="Enterprise-grade RAG with multi-agent orchestration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument for Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Ensure data directory exists
os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Multi-Agent Platform")
    # Initialize any required services here
    yield
    # Shutdown
    logger.info("Shutting down AI Multi-Agent Platform")
    # Clean up resources here

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "vector_db": os.path.exists(VECTOR_DB_PATH),
            "openai": True  # You might want to add a real check here
        }
    }

@app.post("/ingest", response_model=dict)
async def ingest(file: UploadFile):
    """
    Upload and process a document for the knowledge base.
    Supports PDF, TXT, DOCX, and other common formats.
    """
    try:
        # Validate file type
        allowed_types = [".pdf", ".txt", ".docx", ".md"]
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_types)}"
            )

        # Create unique filename to prevent collisions
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_path = f"./app/data/{temp_filename}"
        
        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Process document
        result = ingest_document(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file: {str(e)}")
            
        # Handle processing errors
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result["message"]
            )
            
        return JSONResponse(
            content=result,
            status_code=status.HTTP_201_CREATED
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document ingestion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the document"
        )

@app.get("/query", response_model=dict)
async def ask(question: str, k: int = 4):
    """
    Query the knowledge base using simple RAG
    Parameters:
    - question: The question to answer
    - k: Number of document chunks to retrieve (default 4)
    """
    try:
        if not question or len(question.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question must be at least 3 characters long"
            )
            
        answer = query_rag(question, k)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your question"
        )

@app.get("/multiagent", response_model=dict)
async def multiagent_query(question: str):
    """
    Query the knowledge base using multi-agent system (CrewAI)
    """
    try:
        if not question or len(question.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question must be at least 3 characters long"
            )
            
        answer = run_multiagent_query(question)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-agent query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred in the multi-agent system"
        )

@app.get("/langgraph", response_model=dict)
async def langgraph_query(question: str):
    """
    Query the knowledge base using LangGraph agent orchestration (temporarily disabled)
    """
    return {"answer": "LangGraph functionality is temporarily disabled due to dependency conflicts. Please use the standard RAG or multi-agent endpoints."}

@app.get("/documents")
async def list_documents():
    """
    List all ingested documents in the knowledge base
    """
    try:
        if not os.path.exists(VECTOR_DB_PATH):
            return {"documents": [], "count": 0}
            
        # This is a placeholder - you'd need to implement actual document tracking
        return {
            "documents": ["Document tracking not fully implemented"],
            "count": 1
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve document list"
        )