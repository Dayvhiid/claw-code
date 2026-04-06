import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class OllamaResponse:
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3.5:latest"):
        self.base_url = base_url
        self.model = model

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> OllamaResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                message = result.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
                return OllamaResponse(text=content, tool_calls=tool_calls)
        except urllib.error.URLError as e:
            return OllamaResponse(text=f"Error connecting to Ollama: {e}")
        except Exception as e:
            return OllamaResponse(text=f"Unexpected error: {e}")

if __name__ == "__main__":
    # Test connection
    client = OllamaClient()
    print(f"Testing Ollama client with model: {client.model}")
    resp = client.chat([{"role": "user", "content": "hello"}])
    print(f"Response: {resp.text}")
