import os
from groq import Groq
import requests
import pandas as pd
import streamlit as st
import aisuite as ai
from dotenv import load_dotenv
from typing import Generator
from eod import EodHistoricalData
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")

if "ai_client" not in st.session_state:
    st.session_state.ai_client = ai.Client()

if "eodhd_client" not in st.session_state:
    st.session_state.eodhd_client = EodHistoricalData(EODHD_API_KEY)

st.sidebar.write("Choose the model you want to chat with!")

st.session_state.last_response_required_tool = False

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
    st.session_state.messages = [{
        "role": "system",
        "content": """You are a helpful assistant that uses the tools at your disposal to help users with their queries.
                    Do not use the tools several times, but only use when strictly necessary, with minimal calls.""",
    }]

# show messages history
for message in st.session_state.messages:
    if (message["role"] in ["assistant", "user"]) and ("content" in message):
        if message["content"] != "":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# define function that yields chat responses through streaming
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    st.session_state.last_response_required_tool = False
    for chunk in chat_completion:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
        if delta.tool_calls:
            print(delta)
            st.session_state.last_response_required_tool = True
            st.markdown("Retrieving data...")
            tool_call_list = []
            tool_output_list = []
            # Model required a tool call, so I need to call it with its arguments.
            for tool_call in delta.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                tool_call_args_string = tool_call.function.arguments
                tool_call_args = json.loads(tool_call_args_string)

                tool_call_list.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": 
                            {
                                "name": tool_name,
                                "arguments": tool_call_args_string
                            }
                        })
                
                if tool_name == "get_eod_historical_stock_market_data":
                    tool_output = get_eod_historical_stock_market_data(**tool_call_args)

                tool_output_list.append(
                    {
                        "tool_call_id": tool_call_id, 
                        "role": "tool", # Indicates this message is from tool use
                        "name": tool_name,
                        "content": tool_output,
                    }
                )

            # Add tool call to messages history
            st.session_state.messages.append({
                "role": "assistant",
                "tool_calls": tool_call_list,
            })

            # Add tool output to messages history
            st.session_state.messages.extend(tool_output_list)


# TOOLS DEFINITION
def get_eod_historical_stock_market_data(symbol: str, from_date: str, to_date: str, order: str = 'a', period: str = 'm'):
    result = st.session_state.eodhd_client.get_prices_eod(symbol=symbol, from_=from_date, to=to_date, order=order, period=period, fmt="csv")
    return json.dumps(result)


st.session_state.tools = [
    {
        "type": "function",
        "function": {
            "name": "get_eod_historical_stock_market_data",
            "description": "Retrieve end-of-day historical stock market data for a specific symbol within a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": """
                        The stock symbol or ticker, together with the respective exchange. Some examples:
                        Bitcoin value in USD: 'BTC-USD.CC', where CC stands for cryptocurrency.
                        Apple Inc. stock in NASDAQ: 'AAPL.US'.

                        """
                    },
                    "from_date": {
                        "type": "string",
                        "description": "The start date for historical data in 'YYYY-MM-DD' format."
                    },
                    "to_date": {
                        "type": "string",
                        "description": "The end date for historical data in 'YYYY-MM-DD' format."
                    },
                    "order": {
                        "type": "string",
                        "enum": ["a", "d"],
                        "description": "The order of the data: 'a' for ascending, 'd' for descending.",
                        "default": "a"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["d", "w", "m", "q", "y"],
                        "description": "The data aggregation period: 'd' for daily, 'w' for weekly, 'm' for monthly, 'q' for quarterly, 'y' for yearly.",
                        "default": "m"
                    }
                },
                "required": ["symbol", "from_date", "to_date"]
            }
        }
    }
]


prompt = st.chat_input("Enter a message to the AI")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})


    with st.chat_message("assistant"):
        # send message to Groq API
        stream = st.session_state.ai_client.chat.completions.create(
            model="groq:"+st.session_state.selected_model,
            messages=st.session_state.messages,
            temperature=0.0,
            stream=True,
            tools=st.session_state.tools
        )

        response = st.write_stream(generate_chat_responses(stream))

        st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.last_response_required_tool:
            second_stream = st.session_state.ai_client.chat.completions.create(
                model="groq:"+st.session_state.selected_model,
                messages=st.session_state.messages,
                temperature=0.0,
                stream=True,
                tools=st.session_state.tools,
            )

            second_response = st.write_stream(generate_chat_responses(second_stream))

            st.session_state.messages.append({"role": "assistant", "content": second_response})


        
    

