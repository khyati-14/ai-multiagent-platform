import os
import logging
from typing import Optional, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import OPENAI_API_KEY, OLLAMA_URL, VECTOR_DB_PATH, DEVELOPMENT_MODE, AI_PROVIDER

# Alternative AI imports
try:
    from sentence_transformers import SentenceTransformer
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models based on AI provider
if AI_PROVIDER == "ollama" and OLLAMA_AVAILABLE:
    logger.info("Using Ollama/Llama models")
    try:
        # Use Hugging Face embeddings (more reliable than Ollama embeddings)
        if HUGGINGFACE_AVAILABLE:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        else:
            # Fallback to mock embeddings if Hugging Face not available
            from langchain.embeddings.base import Embeddings
            class MockEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    return [[0.1] * 384 for _ in texts]
                def embed_query(self, text):
                    return [0.1] * 384
            embeddings = MockEmbeddings()
        
        # Use Ollama for LLM with proper configuration
        llm = Ollama(
            base_url=OLLAMA_URL,
            model="llama3.2",
            temperature=0.7
        )
    except Exception as e:
        logger.warning(f"Ollama not available, falling back to mock: {e}")
        # Fallback to mock
        from langchain.embeddings.base import Embeddings
        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 384
        embeddings = MockEmbeddings()
        
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "Ollama service not available. Please ensure Ollama is running and has models installed."
                return MockResponse()
        llm = MockLLM()

elif DEVELOPMENT_MODE:
    # Use free Hugging Face models if available, otherwise mock
    if HUGGINGFACE_AVAILABLE:
        logger.info("Using Hugging Face models (free alternative)")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Simple mock LLM for development
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "This is a free Hugging Face response in development mode. The system is using local embeddings and mock AI responses."
                return MockResponse()
        
        llm = MockLLM()
    else:
        # Fallback to mock embeddings
        from langchain.embeddings.base import Embeddings
        
        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 384
        
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "This is a mock AI response in development mode. Install transformers and sentence-transformers for free Hugging Face models."
                return MockResponse()
        
        embeddings = MockEmbeddings()
        llm = MockLLM()
else:
    # Production mode - try providers in order: OpenAI, Hugging Face, Mock
    if AI_PROVIDER == "openai" and OPENAI_API_KEY:
        logger.info("Using OpenAI models")
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-large"
        )
        llm = ChatOpenAI(
            model="gpt-4-0125-preview",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
    elif HUGGINGFACE_AVAILABLE:
        logger.info("Using Hugging Face models as fallback")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "This is a free Hugging Face response. The system is using local embeddings without API costs."
                return MockResponse()
        
        llm = MockLLM()
    else:
        logger.warning("No AI providers available, using mock responses")
        from langchain.embeddings.base import Embeddings
        
        class MockEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 384
        
        embeddings = MockEmbeddings()
        
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    content = "No AI providers configured. Please set up OpenAI, Ollama, or Hugging Face."
                return MockResponse()
        
        llm = MockLLM()

def get_file_loader(file_path: str):
    """Factory method for file loaders"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return PyPDFLoader(file_path)
    elif file_ext in ('.docx', '.doc'):
        return UnstructuredWordDocumentLoader(file_path)
    elif file_ext == '.txt':
        return TextLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path)

def ingest_document(file_path: str) -> dict:
    """Process and store document with enhanced error handling"""
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
            
        # Load document
        loader = get_file_loader(file_path)
        docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        
        if not chunks:
            raise ValueError("No valid chunks extracted from document")
            
        # Create or update vector store
        try:
            # Always create a new vector store to avoid compatibility issues
            import shutil
            if os.path.exists(VECTOR_DB_PATH):
                if os.path.isdir(VECTOR_DB_PATH):
                    shutil.rmtree(VECTOR_DB_PATH)
                else:
                    os.remove(VECTOR_DB_PATH)
                action = "recreated"
            else:
                action = "created"
            
            # Create fresh vector store
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(VECTOR_DB_PATH)
            
        except Exception as store_error:
            logger.error(f"Error with vector store: {store_error}")
            raise
            
        return {
            "status": "success",
            "action": action,
            "chunks": len(chunks),
            "file": os.path.basename(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "file": os.path.basename(file_path)
        }

def query_rag(question: str, k: int = 4) -> Optional[str]:
    """Enhanced RAG query with fallback"""
    try:
        # Check if knowledge base exists
        if not os.path.exists(VECTOR_DB_PATH):
            return "No documents have been processed yet. Please upload documents first."
            
        # Load vector store with error handling for compatibility issues
        try:
            vector_store = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except KeyError as ke:
            if "__fields_set__" in str(ke):
                logger.warning("Vector store corrupted due to Pydantic version mismatch. Cleaning up.")
                import shutil
                shutil.rmtree(VECTOR_DB_PATH)
                return "Knowledge base was corrupted and has been reset. Please upload documents again."
            else:
                raise
        except Exception as load_error:
            logger.error(f"Could not load vector store for query: {load_error}")
            return "Error loading knowledge base. Please try uploading documents again."
        
        # Retrieve relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        
        if not docs:
            return "No relevant information found in the knowledge base for this question."
            
        # Format context
        context = "\n\n--- DOCUMENT CHUNK ---\n\n".join(
            [f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" 
             for d in docs]
        )
        
        # Generate answer
        prompt = f"""You are an expert knowledge assistant. Answer the user's question 
        based only on the following context. If you don't know the answer, say so.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        response = llm.invoke(prompt)
        # Handle different response types - Ollama returns string directly
        if hasattr(response, 'content'):
            return response.content
        else:
            return response
        
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        return f"An error occurred while processing your request: {str(e)}"

def get_document_chunks(document_path: str) -> List[Document]:
    """Retrieve chunks for a specific document"""
    try:
        if not os.path.exists(VECTOR_DB_PATH):
            return []
            
        try:
            vector_store = FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except KeyError as ke:
            if "__fields_set__" in str(ke):
                logger.warning("Vector store corrupted in get_document_chunks. Cleaning up.")
                import shutil
                shutil.rmtree(VECTOR_DB_PATH)
                return []
            else:
                raise
        except Exception as load_error:
            logger.error(f"Could not load vector store for document chunks: {load_error}")
            return []
        
        # This is simplified - you'd need to implement proper document filtering
        all_chunks = vector_store.docstore._dict.values()
        return [chunk for chunk in all_chunks if document_path in chunk.metadata.get("source", "")]
        
    except Exception as e:
        logger.error(f"Error getting document chunks: {str(e)}", exc_info=True)
        return []