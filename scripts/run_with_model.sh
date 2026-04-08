#!/usr/bin/env bash
# Helper to launch the UI with an explicit MODEL_NAME
# Usage: ./scripts/run_with_model.sh <model-name>

set -euo pipefail
MODEL=${1:-}
if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model-name>"
  exit 1
fi

export MODEL_NAME="$MODEL"
echo "Starting UI with MODEL_NAME=$MODEL_NAME"
python ui_gradio.py
