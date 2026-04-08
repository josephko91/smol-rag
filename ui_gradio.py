import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/chat"

# Simple health check (best-effort)
try:
    requests.get("http://127.0.0.1:8000/status", timeout=1)
except Exception:
    pass


def respond(message, history):
    history = history or []
    history.append({'role':'user', 'content': message})
    try:
        r = requests.post(API_URL, json={"message": message}, timeout=60)
        r.raise_for_status()
        reply = r.json().get("reply", "")
    except Exception as e:
        reply = f"Error: {e}"
    history.append({'role':'assistant', 'content':reply})
    return "", history


with gr.Blocks() as demo:
    chat = gr.Chatbot()
    txt = gr.Textbox(placeholder="Ask something...", show_label=False)
    txt.submit(respond, [txt, chat], [txt, chat])
    txt.submit(lambda: None, None, txt)


demo.launch()
