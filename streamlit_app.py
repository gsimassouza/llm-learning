import os
from groq import Groq
import requests
import pandas as pd
import streamlit as st
import aisuite as ai
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if "client" not in st.session_state:
    st.session_state.client = ai.Client()

st.sidebar.write("Choose the model you want to chat with!")

st.session_state.model_names = [
    # "whisper-large-v3",
    # "whisper-large-v3-turbo",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
    "llama-3.3-70b-specdec",
    "llama-3.2-1b-preview",
    "llama3-70b-8192",
    # "distil-whisper-large-v3-en",
    "llama-3.2-3b-preview",
    "llama3-8b-8192",
    # "llama-guard-3-8b",
    "llama-3.1-8b-instant",
    "llama-3.2-11b-vision-preview"
]

st.session_state.selected_model = st.sidebar.selectbox(label="model", options=st.session_state.model_names)

st.title("Let's chat!")

## Creating the chat

# create list of messages if not existant
if "messages" not in st.session_state:
    st.session_state.messages = []

# show messages history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Enter a message to the AI")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # send message to Groq API
    response = st.session_state.client.chat.completions.create(
        model="groq:"+st.session_state.selected_model,
        messages=st.session_state.messages,
        temperature=0.0
    )
    response_content = response.choices[0].message.content
    
    with st.chat_message("assistant"):
        st.markdown(response_content)
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})
