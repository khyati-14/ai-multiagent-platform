from typing import Optional, Dict
# from crewai import Agent, Task, Crew
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from app.services.rag_pipeline import query_rag
from app.config import OPENAI_API_KEY
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom tool for RAG retrieval
@tool
def retrieve_information(question: str) -> str:
    """Retrieve relevant information from the knowledge base"""
    try:
        return query_rag(question)
    except Exception as e:
        logger.error(f"Error in retrieval tool: {str(e)}", exc_info=True)
        return f"Error retrieving information: {str(e)}"

def create_agents() -> Dict[str, str]:
    """Create and configure the agent team (simplified for compatibility)"""
    return {
        "retriever": "Research Analyst - retrieves information from knowledge base",
        "analyst": "Data Analyst - analyzes retrieved information", 
        "quality": "Quality Assurance - ensures accuracy and completeness"
    }

def run_multiagent_query(question: str) -> str:
    """
    Run a multi-agent query (temporarily using simplified approach)
    """
    try:
        # Temporarily use direct RAG query since CrewAI is disabled
        logger.info("Multi-agent system temporarily using simplified RAG approach")
        result = query_rag(question)
        
        if result:
            return f"[Multi-Agent Mode - Simplified] {result}"
        else:
            return "Multi-agent system is temporarily unavailable. CrewAI dependencies have been disabled due to conflicts."
        
    except Exception as e:
        logger.error(f"Error in multi-agent query: {str(e)}", exc_info=True)
        return f"Multi-agent system encountered an error: {str(e)}"