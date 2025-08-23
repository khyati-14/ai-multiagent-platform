import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/app/data/vectorstore")
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()  # openai, ollama, huggingface, mock

# Validation based on AI provider
if AI_PROVIDER == "openai" and not OPENAI_API_KEY and not DEVELOPMENT_MODE:
    raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")

VECTOR_DB_PATH = "app/data/vectorstore"
