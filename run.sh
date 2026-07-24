#!/bin/bash
# Run a script with the gpcrowdkit venv and GPU libraries configured.
# Usage: ./run.sh script.py [args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
NVIDIA_BASE="$VENV/lib/python3.10/site-packages/nvidia"

export LD_LIBRARY_PATH=$(find "$NVIDIA_BASE" -maxdepth 2 -name "lib" -type d | tr '\n' ':')
export TF_USE_LEGACY_KERAS=1

exec "$VENV/bin/python" "$@"
