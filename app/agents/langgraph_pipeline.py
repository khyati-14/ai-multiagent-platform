from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from app.services.rag_pipeline import query_rag
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY
import logging
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    model="gpt-4-0125-preview",
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY
)

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    messages: list
    iterations: int

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant context from knowledge base"""
    try:
        state["context"] = query_rag(state["question"])
        state["messages"].append(
            AIMessage(content=f"Retrieved context: {state['context'][:200]}...")
        )
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}", exc_info=True)
        state["context"] = f"Error retrieving context: {str(e)}"
    return state

def analyze_and_answer(state: AgentState) -> AgentState:
    """Analyze context and generate answer"""
    try:
        prompt = [
            {"role": "system", "content": "You are an expert analyst. Answer the question based on the context."},
            {"role": "user", "content": f"Context:\n{state['context']}\n\nQuestion: {state['question']}"}
        ]
        
        response = llm.invoke(prompt)
        # Handle different response types - Ollama returns string directly
        if hasattr(response, 'content'):
            state["answer"] = response.content
        else:
            state["answer"] = response
        state["messages"].append(AIMessage(content=f"Draft answer: {state['answer'][:200]}..."))
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        state["answer"] = f"Error generating answer: {str(e)}"
    return state

def quality_check(state: AgentState) -> AgentState:
    """Verify the answer meets quality standards"""
    try:
        if "error" in state["answer"].lower():
            return state
            
        prompt = [
            {"role": "system", "content": "You are a quality assurance specialist. Review the answer."},
            {"role": "user", "content": f"Question: {state['question']}\n\nProposed Answer: {state['answer']}\n\nIs this answer complete and accurate? If not, suggest improvements."}
        ]
        
        response = llm.invoke(prompt)
        # Handle different response types - Ollama returns string directly
        response_text = response.content if hasattr(response, 'content') else response
        state["messages"].append(AIMessage(content=f"QA feedback: {response_text[:200]}..."))
        
        # If QA suggests improvements, we loop back
        if "improve" in response_text.lower() or "incomplete" in response_text.lower():
            if state["iterations"] < 3:  # Max 3 iterations
                state["iterations"] += 1
                state["answer"] = ""  # Clear to trigger reprocessing
            else:
                state["answer"] += "\n\n[Note: QA process reached maximum iterations]"
    except Exception as e:
        logger.error(f"Error in QA: {str(e)}", exc_info=True)
    return state

def should_continue(state: AgentState) -> str:
    """Determine if we should continue processing"""
    if not state.get("answer"):
        return "analyze"
    if state["iterations"] >= 3:
        return END
    if "improve" in state["messages"][-1].content.lower():
        return "analyze"
    return END

# Build the workflow
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("retriever", retrieve_context)
workflow.add_node("analyzer", analyze_and_answer)
workflow.add_node("qa", quality_check)

# Define edges
workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "analyzer")
workflow.add_edge("analyzer", "qa")
workflow.add_conditional_edges(
    "qa",
    should_continue,
    {
        "analyze": "analyzer",
        END: END
    }
)

# Compile the workflow
langgraph_agent = workflow.compile()

def run_langgraph_query(question: str) -> str:
    """Execute a query using LangGraph orchestration"""
    try:
        initial_state = {
            "question": question,
            "context": "",
            "answer": "",
            "messages": [HumanMessage(content=question)],
            "iterations": 0
        }
        
        result = langgraph_agent.invoke(initial_state)
        
        # Format the final answer
        final_answer = f"""## Answer\n{result['answer']}\n\n## Process Summary"""
        for msg in result["messages"]:
            final_answer += f"\n\n- {msg.type}: {msg.content[:200]}..."
            
        return final_answer
        
    except Exception as e:
        logger.error(f"Error in LangGraph query: {str(e)}", exc_info=True)
        return f"An error occurred in the LangGraph workflow: {str(e)}"