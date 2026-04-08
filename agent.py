"""Research agent using LangChain RAG with Ollama + Chroma"""
import os
import threading
import logging
import tiktoken

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from retriever import Retriever
from config import OLLAMA_BASE_URL, MODEL_NAME, MODEL_TEMPERATURE, MODEL_MAX_TOKENS, MAX_PROMPT_TOKENS

# Configure logging for debugging RAG context
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer for token counting (using cl100k_base works well for most models)
try:
    token_encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    # Fallback: use simple approximation if tiktoken fails
    token_encoder = None


def format_docs(docs):
    """Format retrieved documents for prompt with token-aware truncation.
    
    Instead of character-based limiting, this now uses token counting to respect
    actual token consumption. Ensures we don't accidentally exceed context window.
    """
    parts = []
    total_tokens = 0
    token_budget = MAX_PROMPT_TOKENS if token_encoder else None
    
    for d in docs:
        text = (d.get('text') or '').strip()
        if not text:
            continue
        # skip very short noise
        if len(text) < 50:
            continue
        
        # Count tokens in this piece
        if token_encoder and token_budget:
            piece_tokens = len(token_encoder.encode(text))
            # If this piece alone would exceed budget, skip it
            if piece_tokens > token_budget - total_tokens:
                logger.debug(f"Skipping doc (tokens {piece_tokens} > remaining {token_budget - total_tokens})")
                break
            total_tokens += piece_tokens
        else:
            # Fallback: approximate 1 token per 4 characters
            piece_tokens = len(text) // 4
            if piece_tokens > token_budget - total_tokens if token_budget else False:
                break
            total_tokens += piece_tokens
        
        parts.append(text)
    
    ctx = "\n\n".join(parts)
    
    if token_encoder:
        actual_tokens = len(token_encoder.encode(ctx))
        logger.info(f"Formatted context: {actual_tokens} tokens, {len(ctx)} characters")
    else:
        logger.info(f"Formatted context: ~{total_tokens} est. tokens, {len(ctx)} characters")
    
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
        
        # Conversation memory: list of (user_question, assistant_answer) tuples
        # Kept in memory for this session. Older turns are pruned to save tokens.
        self.conversation_history = []
        self.max_history_items = 5  # Keep last 5 turns
        
        # Define prompt template for RAG (with context and conversation history)
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are an expert research assistant. Use the provided context and conversation history to answer questions thoroughly and accurately.
If the answer is not contained in the context, clearly state that you don't have enough information.

{history}

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Define prompt template for direct conversation (no context, but with history)
        self.simple_prompt = ChatPromptTemplate.from_template(
            """You are a helpful and friendly assistant. Answer the user's question naturally and conversationally.

{history}

Question: {question}

Answer:"""
        )
        
        # Build RAG chain (with retrieval)
        self.rag_chain = (
            {
                "context": lambda x: format_docs(self.retriever_obj.retrieve(x["question"])),
                "question": RunnablePassthrough(),
                "history": lambda x: self._format_history(),
            }
            | self.rag_prompt
            | self.llm
        )
        
        # Build simple chain (without retrieval)
        self.simple_chain = (
            {
                "history": lambda x: self._format_history(),
                "question": RunnablePassthrough(),
            }
            | self.simple_prompt
            | self.llm
        )

        # Warm up the model/runtime once in a background thread to avoid
        # a slow first interactive request. Controlled by the WARMUP env var
        # (set to "0" to disable).
        try:
            if os.environ.get("WARMUP", "1") != "0":
                def _warm():
                    try:
                        # small, friendly prompt to initialize tokenizer/graph/metal kernels
                        self.simple_chain.invoke({"question": "Hello"})
                    except Exception:
                        pass
                threading.Thread(target=_warm, daemon=True).start()
        except Exception:
            pass
    
    def _format_history(self) -> str:
        """Format conversation history for inclusion in prompt.
        
        Returns a formatted string with recent conversation turns, or empty string if no history.
        """
        if not self.conversation_history:
            return ""
        
        history_lines = ["Recent conversation history:"]
        for user_q, assistant_a in self.conversation_history:
            # Truncate very long answers to keep history manageable
            answer_preview = assistant_a[:300] + "..." if len(assistant_a) > 300 else assistant_a
            history_lines.append(f"User: {user_q}")
            history_lines.append(f"Assistant: {answer_preview}")
        
        return "\n".join(history_lines) + "\n"
    
    def _add_to_history(self, question: str, answer: str):
        """Add a Q&A pair to conversation history, keeping recent items only."""
        self.conversation_history.append((question, answer))
        # Prune old history if we exceed max_history_items
        if len(self.conversation_history) > self.max_history_items:
            self.conversation_history.pop(0)
            logger.debug(f"Pruned conversation history to {len(self.conversation_history)} items")
        else:
            logger.info(f"Conversation history: {len(self.conversation_history)} items")
    
    def clear_history(self):
        """Clear conversation history to start fresh."""
        old_len = len(self.conversation_history)
        self.conversation_history.clear()
        logger.info(f"Cleared {old_len} items from conversation history")
    
    def answer(self, question: str, use_rag: bool | None = None) -> str:
        """Answer a question intelligently with conversation memory.

        If `use_rag` is None, the agent will decide automatically whether to
        retrieve documents (based on `needs_retrieval`). If `use_rag` is True/False
        the behavior is forced accordingly.
        
        Maintains conversation history to provide context across multiple turns.
        """
        try:
            # Determine whether to use retrieval: explicit flag wins, otherwise auto
            if use_rag is None:
                retrieval_needed = needs_retrieval(question)
            else:
                retrieval_needed = bool(use_rag)

            if retrieval_needed:
                # Use RAG chain with document retrieval
                logger.info(f"Using RAG for question: {question[:100]}...")
                docs = self.retriever_obj.retrieve(question)
                logger.info(f"Retrieved {len(docs)} documents")
                for i, doc in enumerate(docs[:3]):  # Log first 3 docs' metadata
                    meta = doc.get('meta', {})
                    text_preview = doc.get('text', '')[:100].replace('\n', ' ')
                    logger.info(f"  Doc {i}: {meta} | Preview: {text_preview}...")
                
                context = format_docs(docs)
                logger.info(f"Formatted context length: {len(context)} characters")
                if len(context) > 500:
                    logger.info(f"Context preview (first 500 chars): {context[:500]}...")
                
                response = self.rag_chain.invoke({"question": question})
            else:
                # Use simple chain without retrieval for conversational queries
                logger.info(f"Using simple (no-RAG) mode for: {question[:100]}...")
                response = self.simple_chain.invoke({"question": question})

            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            # Add to conversation history for follow-up context
            self._add_to_history(question, answer_text)
            
            return answer_text
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return f"Error generating answer: {e}"


if __name__ == '__main__':
    agent = ResearchAgent()
    q = "Summarize the key findings in the provided documents"
    print(agent.answer(q))
