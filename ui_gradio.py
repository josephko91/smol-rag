import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/chat"

# Simple health check (best-effort)
try:
    requests.get("http://127.0.0.1:8000/status", timeout=1)
except Exception:
    pass


def respond(message, history, force_rag=False):
    """
    Send message to agent.
    
    - If force_rag is unchecked (False): agent auto-detects via needs_retrieval()
      (greetings won't use RAG, domain questions will)
    - If force_rag is checked (True): always use RAG regardless of question type
    """
    history = history or []
    history.append({'role': 'user', 'content': message})
    try:
        payload = {"message": message}
        # Only pass use_rag if forcing it; otherwise let agent auto-detect
        if force_rag:
            payload["use_rag"] = True  # Force RAG
        # else: omit use_rag, agent will auto-detect via needs_retrieval()
        r = requests.post(API_URL, json=payload, timeout=60)
        r.raise_for_status()
        reply = r.json().get("reply", "")
    except Exception as e:
        reply = f"Error: {e}"
    history.append({'role': 'assistant', 'content': reply})
    return "", history


with gr.Blocks() as demo:
    chat = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(placeholder="Ask something...", show_label=False)
        force_rag = gr.Checkbox(label="Force RAG (skip auto-detection)", value=False)

    # Submit the message with the optional force_rag flag
    txt.submit(respond, [txt, chat, force_rag], [txt, chat])
    txt.submit(lambda: None, None, txt)


demo.launch()
