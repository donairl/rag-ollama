import json
import base64
import streamlit as st

def load_chat_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_chat_history(history):
    with open("chat_history.json", "w") as f:
        json.dump(history, f)

def get_download_link(chat_history):
    markdown_text = "# Chat History\n\n"
    for message in chat_history:
        markdown_text += f"**{message['role'].capitalize()}:** {message['content']}\n\n"
    
    b64 = base64.b64encode(markdown_text.encode()).decode()
    href = f'<a href="data:file/markdown;base64,{b64}" download="chat_history.md">Download Chat History</a>'
    return href