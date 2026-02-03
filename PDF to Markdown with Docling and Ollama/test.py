import requests
import json

OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "qwen3-vl:2b"

# Test 1: OpenAI-compatible endpoint
print("=== Testing OpenAI-compatible endpoint ===")
try:
    response = requests.post(
        f"{OLLAMA_URL}/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe this: A simple test"
                }
            ],
            "max_tokens": 100
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    print(f"Response type: {type(response.json())}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Testing Ollama native endpoint ===")
try:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": VLM_MODEL,
            "prompt": "Describe this: A simple test",
            "stream": False
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    print(f"Response type: {type(response.json())}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Testing Ollama chat endpoint ===")
try:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe this: A simple test"
                }
            ],
            "stream": False
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    print(f"Response type: {type(response.json())}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")