"""Research agent using LangChain RAG with Ollama + Chroma"""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from retriever import Retriever
from config import OLLAMA_BASE_URL, MODEL_NAME, MODEL_TEMPERATURE, MODEL_MAX_TOKENS, MAX_PROMPT_TOKENS


def format_docs(docs):
    """Format retrieved documents for prompt"""
    # Build a context string but cap its size to avoid very large prompts
    # which increase latency. MAX_PROMPT_TOKENS is a character-based cap
    # in this simple implementation (approximate).
    parts = []
    total = 0
    for d in docs:
        text = (d.get('text') or '').strip()
        if not text:
            continue
        # skip very short noise
        if len(text) < 50:
            continue
        parts.append(text)
        total += len(text)
        if total >= MAX_PROMPT_TOKENS:
            break
    ctx = "\n\n".join(parts)
    if len(ctx) > MAX_PROMPT_TOKENS:
        return ctx[:MAX_PROMPT_TOKENS].rsplit('\n', 1)[0] + "\n\n..."
    return ctx


def needs_retrieval(question: str) -> bool:
    """Determine if a query needs document retrieval.
    
    Returns False for greetings and general conversation.
    Returns True for domain-specific or complex queries.
    """
    question_lower = question.lower().strip()
    
    # Greetings and conversational phrases that don't need retrieval
    conversational_keywords = [
        'hello', 'hi ', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how\'s it going', 'what\'s up', 'thanks', 'thank you',
        'please', 'sorry', 'excuse me', 'bye', 'goodbye', 'see you',
        'what is your name', 'who are you', 'tell me about yourself'
    ]
    
    for keyword in conversational_keywords:
        if keyword in question_lower:
            return False
    
    # Also return False if question is very short (likely greeting/filler)
    if len(question.split()) <= 2:
        return False
    
    return True


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
        
        # Define prompt template for RAG (with context)
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are an expert research assistant. Use the provided context to answer the question thoroughly and accurately.
If the answer is not contained in the context, clearly state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Define prompt template for direct conversation (no context)
        self.simple_prompt = ChatPromptTemplate.from_template(
            """You are a helpful and friendly assistant. Answer the user's question naturally and conversationally.

Question: {question}

Answer:"""
        )
        
        # Build RAG chain (with retrieval)
        self.rag_chain = (
            {
                "context": lambda x: format_docs(self.retriever_obj.retrieve(x["question"])),
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
        )
        
        # Build simple chain (without retrieval)
        self.simple_chain = (
            self.simple_prompt
            | self.llm
        )
    
    def answer(self, question: str, use_rag: bool | None = None) -> str:
        """Answer a question intelligently.

        If `use_rag` is None, the agent will decide automatically whether to
        retrieve documents (based on `needs_retrieval`). If `use_rag` is True/False
        the behavior is forced accordingly.
        """
        try:
            # Determine whether to use retrieval: explicit flag wins, otherwise auto
            if use_rag is None:
                retrieval_needed = needs_retrieval(question)
            else:
                retrieval_needed = bool(use_rag)

            if retrieval_needed:
                # Use RAG chain with document retrieval
                response = self.rag_chain.invoke({"question": question})
            else:
                # Use simple chain without retrieval for conversational queries
                response = self.simple_chain.invoke({"question": question})

            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating answer: {e}"


if __name__ == '__main__':
    agent = ResearchAgent()
    q = "Summarize the key findings in the provided documents"
    print(agent.answer(q))
