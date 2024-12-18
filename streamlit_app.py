import os
from groq import Groq
import requests
import pandas as pd

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(pd.DataFrame(response.json()['data']))

