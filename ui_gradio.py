import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/chat"

# Simple health check (best-effort)
try:
    requests.get("http://127.0.0.1:8000/status", timeout=1)
except Exception:
    pass


def respond(message, history, use_rag=True):
    history = history or []
    history.append({'role': 'user', 'content': message})
    try:
        payload = {"message": message}
        # include flag only if explicitly set (bool) to keep default semantics
        payload["use_rag"] = bool(use_rag)
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
        use_rag = gr.Checkbox(label="Use RAG (retrieve documents)", value=True)

    # Submit the message with the selected `use_rag` option
    txt.submit(respond, [txt, chat, use_rag], [txt, chat])
    txt.submit(lambda: None, None, txt)


demo.launch()
