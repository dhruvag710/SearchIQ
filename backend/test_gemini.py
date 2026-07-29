from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(".env"))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

model = "gemini-2.0-flash"

print(f"Testing model: {model}")

response = client.models.generate_content(
    model=model,
    contents="Reply with exactly: Hello SearchIQ"
)

print(response.text)