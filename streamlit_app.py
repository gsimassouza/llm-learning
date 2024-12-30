import os
from groq import Groq
import requests
import pandas as pd
import streamlit as st

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

st.sidebar.write("Choose the model you want to talk to!")

st.session_state.model_names = [
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "mixtral-8x7b-32768",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
    "llama-3.3-70b-specdec",
    "llama-3.2-1b-preview",
    "llama3-70b-8192",
    "distil-whisper-large-v3-en",
    "llama-3.2-3b-preview",
    "llama3-8b-8192",
    "llama-guard-3-8b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-11b-vision-preview"
]

st.sidebar.selectbox(label="model", options=st.session_state.model_names)

st.title("Bora conversar!")

# To-do: create the chat itself

