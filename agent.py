"""Agent that composes retrieved context and queries the LLM backend"""
from model_backends import LLMBackend
from retriever import Retriever
from config import LLAMA_MODEL_PATH


class ResearchAgent:
    def __init__(self, model_path: str = LLAMA_MODEL_PATH):
        self.retriever = Retriever()
        self.llm = LLMBackend(model_path=model_path)

    def answer(self, query: str, top_k: int = 5) -> str:
        docs = self.retriever.retrieve(query, k=top_k)
        context = "\n\n".join([d['text'] for d in docs])
        prompt = f"Use the following context to answer the question. If the answer is not contained, say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return self.llm.generate(prompt)


if __name__ == '__main__':
    agent = ResearchAgent()
    q = "What is the primary contribution of the provided documents?"
    print(agent.answer(q))
