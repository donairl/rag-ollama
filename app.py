import streamlit as st
import requests
import json
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import tempfile
import os

# Utility functions
from utils import load_chat_history, save_chat_history, get_download_link

# Set page config
st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar
st.sidebar.title("Chatbot Settings")


#get list of model from ollama api  
response = requests.get("http://localhost:11434/api/tags")
models = response.json()
model = st.sidebar.selectbox("Select Model", models)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.number_input("Max Tokens", 10, 2000, 500, 10)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a file for RAG", type=["pdf", "txt"])
if uploaded_file:
    file_type = uploaded_file.type
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if file_type == "application/pdf":
        loader = PyPDFLoader(temp_file_path)
    else:
        loader = TextLoader(temp_file_path)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    st.session_state.vector_store = FAISS.from_documents(texts, embeddings)

    os.unlink(temp_file_path)
    st.sidebar.success("File uploaded and processed for RAG.")

# Main chat interface
st.title("Ollama Chatbot")
st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": user_input,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # If RAG is enabled, add context from vector store
        if st.session_state.vector_store:
            context = st.session_state.vector_store.similarity_search(user_input, k=3)
            context_text = "\n".join([doc.page_content for doc in context])
            payload["prompt"] = f"Context: {context_text}\n\nQuestion: {user_input}"

        # Make a POST request to the Ollama API
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_response += chunk['response']
                    message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# Save chat history
save_chat_history(st.session_state.chat_history)

# Download chat history
st.sidebar.markdown(get_download_link(st.session_state.chat_history), unsafe_allow_html=True)

# Clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    save_chat_history([])
    st.experimental_rerun()