"""Research agent using LangChain RAG with Ollama + Chroma"""
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from retriever import Retriever
from config import OLLAMA_BASE_URL, MODEL_NAME, MODEL_TEMPERATURE, MODEL_MAX_TOKENS, MAX_PROMPT_TOKENS


def format_docs(docs):
    """Format retrieved documents for prompt"""
    return "\n\n".join([d['text'] for d in docs])


class ResearchAgent:
    def __init__(self):
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            num_predict=MODEL_MAX_TOKENS,
        )
        
        # Initialize retriever
        self.retriever_obj = Retriever()
        
        # Define prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert research assistant. Use the provided context to answer the question thoroughly and accurately.
If the answer is not contained in the context, clearly state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Build RAG chain
        self.chain = (
            {
                "context": lambda x: format_docs(self.retriever_obj.retrieve(x["question"])),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )
    
    def answer(self, question: str) -> str:
        """Answer a question using RAG"""
        try:
            response = self.chain.invoke({"question": question})
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating answer: {e}"


if __name__ == '__main__':
    agent = ResearchAgent()
    q = "Summarize the key findings in the provided documents"
    print(agent.answer(q))
