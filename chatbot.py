import requests
import json

OLLAMA_API_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'deepseek-r1:latest'

def clean_json_response(response_text):
    try:
       
        if "```json" in response_text:
            response_text = response_text.split("```json")[-1].split("```")[0]

        elif "```" in response_text:
            response_text = response_text.split("```")[1]
        response_text = response_text.strip()
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        print("Raw response was:", response_text)
        return None

def answer_query(question: str) -> str:
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": True
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_text = response.text
        data = clean_json_response(response_text)
        if data is None:
            return "Sorry, I couldn't parse the response."

        answer = data['results'][0]['text'].strip()
        return answer

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"
