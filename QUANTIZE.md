Running quantized/Metal-optimized models on Apple Silicon (M1/M2)

Overview

This project uses Ollama as the local LLM runner. To get best performance on
Apple Silicon (M1/M2/M2 Pro/Max), use a Metal/MPS-optimized or quantized build
of the model. Ollama frequently publishes quantized variants (names vary by
model/provider), for example tags with `-metal`, `-q4_*`, or `-ggml-metal`.

Quick steps

1. Check available models in Ollama:

```bash
ollama ls
```

Look for model names that indicate quantized or Metal builds (examples: `mistral:7b-q4_0`, `gemma3:4b-metal`). If you don't see a quantized variant, check the model provider or Ollama docs for a quantized release.

2. Start the UI or server with the quantized model by setting `MODEL_NAME`:

```bash
# replace <quant-model> with the model name you want to use
export MODEL_NAME="<quant-model>"
python ui_gradio.py
```

The codebase reads `MODEL_NAME` from the environment (see `config.py`).

3. Verify Metal usage (logs & powermetrics):

- Start Ollama with debug logs:

```bash
# Stop Ollama.app if running; then run:
OLLAMA_LOG=debug ollama serve
# or run a single model interactively
OLLAMA_LOG=debug ollama run <quant-model> --interactive
```

- Trigger an inference and watch logs for `Metal`, `mps`, `offloading`, or `picking default device` messages.
- Optionally monitor `powermetrics` and Activity Monitor → GPU History while sending an inference.

Helper script

There is a small helper script in `scripts/run_with_model.sh` which accepts a model name and launches the UI with that model set as `MODEL_NAME`:

```bash
scripts/run_with_model.sh mistral:7b-q4_0
```

Notes

- Quantized model names vary by provider; using `ollama ls` is the easiest way to find available variants locally.
- If you must download a quantized build, follow the model provider's instructions or Ollama documentation.
- Smaller or quantized models reduce memory and latency but may slightly affect quality.

If you tell me the exact quantized model name you want to use (from `ollama ls`), I can help you measure latency and adjust `MODEL_MAX_TOKENS` / `MODEL_TEMPERATURE` for best throughput.
