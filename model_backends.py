"""Minimal LLM backend abstraction using llama-cpp-python if available"""
from typing import Optional

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


class LLMBackend:
    def __init__(self, model_path: Optional[str] = None):
        if Llama is None:
            raise RuntimeError('llama_cpp not available; install llama-cpp-python or provide another backend')
        self.model = Llama(model_path=model_path)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self.model.create(prompt=prompt, max_tokens=max_tokens)
        return resp.get('choices', [{}])[0].get('text', '').strip()


if __name__ == '__main__':
    print('Model backend loaded (if llama-cpp-python installed)')
