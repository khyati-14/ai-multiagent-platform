import streamlit as st
import requests
import time
import os
from typing import Optional
from streamlit.components.v1 import html
from datetime import datetime

# Custom CSS for premium look
def inject_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4a6fa5;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #3a5a80;
            transform: scale(1.05);
        }
        .stFileUploader>div>div>div>div {
            padding: 20px;
            border: 2px dashed #4a6fa5;
            border-radius: 8px;
            background-color: rgba(74, 111, 165, 0.05);
        }
        .agent-response {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            background-color: #e9f5ff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
            padding: 1.5rem;
        }
        .stMarkdown h1 {
            color: #2c3e50;
            border-bottom: 2px solid #4a6fa5;
            padding-bottom: 0.3em;
        }
        .stMarkdown h2 {
            color: #34495e;
        }
        .user-message {
            background-color: #4a6fa5;
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 80%;
            align-self: flex-end;
        }
        .assistant-message {
            background-color: #e9f5ff;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 80%;
            align-self: flex-start;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "api_mode" not in st.session_state:
        st.session_state.api_mode = "Standard RAG"
    if "processing" not in st.session_state:
        st.session_state.processing = False

# API interaction helper
def call_api(endpoint: str, params: Optional[dict] = None, files: Optional[dict] = None):
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    try:
        if files:
            response = requests.post(f"{API_URL}{endpoint}", files=files)
        else:
            response = requests.get(f"{API_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Display conversation history
def display_conversation():
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-message'>{msg['content']}</div>", 
                unsafe_allow_html=True
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                
                if msg.get("sources"):
                    with st.expander("View sources"):
                        st.markdown(msg["sources"])
                
                if msg.get("processing_time"):
                    st.caption(f"Processed in {msg['processing_time']:.2f}s")

# Document processing section
def document_processing():
    st.sidebar.header("Document Management")
    st.sidebar.markdown("Upload documents to build your knowledge base")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed"
    )
    
    if uploaded_file and st.sidebar.button("Process Document"):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            start_time = time.time()
            result = call_api(
                "/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())}
            )
            processing_time = time.time() - start_time
            
            if result and result.get("status") == "success":
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.sidebar.success(
                    f"Processed {uploaded_file.name} in {processing_time:.2f}s "
                    f"({result['chunks']} chunks)"
                )
                st.balloons()
            else:
                if result is None:
                    st.sidebar.error("No response from API. Please check if the FastAPI service is running.")
                else:
                    st.sidebar.error(result.get("message", "Failed to process document"))


# Main chat interface
def chat_interface():
    st.header("Knowledge Assistant")
    
    # Display conversation
    display_conversation()
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.conversation.append({
            "role": "user", 
            "content": prompt
        })
        
        # Immediately show user message
        st.markdown(
            f"<div class='user-message'>{prompt}</div>", 
            unsafe_allow_html=True
        )
        
        # Process with AI
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            if st.session_state.api_mode == "Standard RAG":
                response = call_api("/query", {"question": prompt})
            elif st.session_state.api_mode == "Multi-Agent":
                response = call_api("/multiagent", {"question": prompt})
            else:
                response = call_api("/langgraph", {"question": prompt})
                
            processing_time = time.time() - start_time
            
            if response:
                answer = response.get("answer", "No answer returned")
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": answer,
                    "processing_time": processing_time
                })
                
                # Show assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    st.caption(f"Processed in {processing_time:.2f}s")
            else:
                st.error("Failed to get response from the AI system")

# Settings panel
def settings_panel():
    st.sidebar.header("Settings")
    st.session_state.api_mode = st.sidebar.selectbox(
        "Processing Mode",
        ["Standard RAG", "Multi-Agent", "LangGraph Orchestration"],
        index=0,
        help="Choose how the system processes your queries"
    )
    
    temperature = st.sidebar.slider(
        "Creativity Level",
        0.0, 1.0, 0.3,
        help="Higher values produce more creative responses"
    )
    
    if st.sidebar.button("Clear Conversation"):
        st.session_state.conversation = []
        st.rerun()

# Document library
def document_library():
    st.sidebar.header("Document Library")
    if st.session_state.uploaded_files:
        for doc in st.session_state.uploaded_files:
            st.sidebar.markdown(f"📄 {doc}")
            
        if st.sidebar.button("View Document Chunks"):
            st.session_state.show_chunks = True
    else:
        st.sidebar.info("No documents uploaded yet")

# Main app
def main():
    st.set_page_config(
        page_title="Enterprise Knowledge Assistant",
        page_icon="💡",
        layout="wide"
    )
    
    inject_custom_css()
    init_session()
    
    # App header
    st.title("💡 Enterprise Knowledge Assistant")
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h3>AI-powered document analysis with multi-agent orchestration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    document_processing()
    settings_panel()
    document_library()
    chat_interface()

if __name__ == "__main__":
    main()