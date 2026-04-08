"""LLM backend abstraction using Ollama HTTP API"""
import requests
from config import OLLAMA_BASE_URL, MODEL_NAME, MODEL_MAX_TOKENS, MODEL_TEMPERATURE


class OllamaBackend:
    """Ollama inference backend via HTTP API"""
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_name: str = MODEL_NAME):
        self.base_url = base_url
        self.model_name = model_name
        self.endpoint = f"{base_url}/api/generate"
        
    def generate(self, prompt: str, max_tokens: int = MODEL_MAX_TOKENS, temperature: float = MODEL_TEMPERATURE) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            raise RuntimeError(f"Ollama backend error: {e}")


class LLMBackend(OllamaBackend):
    """Alias for backward compatibility"""
    pass


if __name__ == '__main__':
    backend = OllamaBackend()
    print(f'Ollama backend initialized (endpoint: {backend.endpoint}, model: {backend.model_name})')
