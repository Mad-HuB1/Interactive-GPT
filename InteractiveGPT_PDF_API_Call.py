"""
This Streamlit app provides an interactive chat interface 
for asking questions about a PDF document. 

Users can upload a PDF, configure API settings, 
and chat with an LLM (specifically using a Databricks model endpoint) 
to get answers based on the document's content. 
"""

import streamlit as st
from openai import OpenAI
import os
from typing import List, Dict
from PyPDF2 import PdfReader
import tempfile

# App configuration
st.set_page_config(page_title="Interactive Chat Interface", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded Document."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file.seek(0)
        
        pdf_reader = PdfReader(tmp_file.name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    os.unlink(tmp_file.name)
    return text

def initialize_client(token: str, base_url: str) -> OpenAI:
    """Initialize OpenAI client."""
    return OpenAI(api_key=token, base_url=base_url)

def get_gpt_response(client: OpenAI, messages: List[Dict[str, str]], context: str, model: str) -> str:
    """Get response from GPT model with PDF context."""
    try:
        system_message = f"""You are an AI assistant. Use the following context from the PDF to answer questions:
        {context}
        Only use information from the provided context. If the question cannot be answered from the context, say so."""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                *messages
            ],
            model=model,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("📚 Interactive Chat Interface")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        token = st.text_input("Databricks Token", type="password")
        base_url = st.text_input("Base URL", "https://abc123.azuredatabricks.net/serving-endpoints")
        model = st.text_input("Model Name", "databricks-dbrx-instruct")
        
        uploaded_file = st.file_uploader("Upload Document", type="pdf")
        if uploaded_file:
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            st.success("Document processed successfully!")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    if not token:
        st.warning("Please enter your Databricks token in the sidebar.")
        return

    if not st.session_state.pdf_text:
        st.info("Please upload a Document to begin.")
        return

    # Initialize client
    client = initialize_client(token, base_url)

    # Chat interface
    user_input = st.chat_input("Ask about the document...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = get_gpt_response(client, st.session_state.messages, st.session_state.pdf_text, model)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

if __name__ == "__main__":
    main()