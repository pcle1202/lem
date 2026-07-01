import os
import time
import requests
from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()

_groq_client = None
_gemini_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client


def query_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> dict:
    start = time.time()
    try:
        response = _get_groq_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        latency = time.time() - start
        return {
            "text": response.choices[0].message.content,
            "latency": round(latency, 3),
            "error": None
        }
    except Exception as e:
        return {"text": None, "latency": None, "error": str(e)}

def query_gemini(prompt: str, model: str = "gemini-2.5-flash") -> dict:
    start = time.time()
    try:
        response = _get_gemini_client().models.generate_content(
            model=model,
            contents=prompt
        )
        latency = time.time() - start
        return {
            "text": response.text,
            "latency": round(latency, 3),
            "error": None
        }
    except Exception as e:
        return {"text": None, "latency": None, "error": str(e)}

def query_ollama(prompt: str, model: str = "llama3.2:3b") -> dict:
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()
        latency = time.time() - start
        return {
            "text": response.json()["message"]["content"],
            "latency": round(latency, 3),
            "error": None,
        }
    except Exception as e:
        return {"text": None, "latency": None, "error": str(e)}


MODELS = {
    "groq": query_groq,
    "gemini": query_gemini,
    "ollama": query_ollama,
}