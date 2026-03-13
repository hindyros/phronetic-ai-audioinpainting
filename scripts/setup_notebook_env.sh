#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

ENV_DIR="$HOME/.virtualenvs/global_notebooker"
ENV_PYTHON="$ENV_DIR/bin/python"
KERNEL_NAME="global_notebooker"
DISPLAY_NAME="global_notebooker"
CACHE_DIR="$ROOT_DIR/.cache"
MPL_CACHE_DIR="$CACHE_DIR/matplotlib"
FONTCONFIG_CACHE_DIR="$CACHE_DIR/fontconfig"

mkdir -p "$MPL_CACHE_DIR" "$FONTCONFIG_CACHE_DIR"

mkdir -p "$HOME/.virtualenvs"
if [ ! -x "$ENV_PYTHON" ]; then
  uv venv "$ENV_DIR" --python 3.11
fi
uv pip install --python "$ENV_PYTHON" -e ".[dev,notebooks]"
"$ENV_PYTHON" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

export KERNEL_NAME CACHE_DIR MPL_CACHE_DIR ENV_PYTHON
KERNEL_DIR="$("$ENV_PYTHON" -c 'import os; from jupyter_client.kernelspec import KernelSpecManager; print(KernelSpecManager().find_kernel_specs()[os.environ["KERNEL_NAME"]])')"
export KERNEL_DIR
"$ENV_PYTHON" -c 'import json, os, pathlib; path = pathlib.Path(os.environ["KERNEL_DIR"]) / "kernel.json"; data = json.loads(path.read_text()); env = data.setdefault("env", {}); env["MPLCONFIGDIR"] = os.environ["MPL_CACHE_DIR"]; env["XDG_CACHE_HOME"] = os.environ["CACHE_DIR"]; path.write_text(json.dumps(data, indent=1) + "\n")'

echo
echo "Notebook environment is ready."
echo "Interpreter: $ENV_PYTHON"
echo "Kernel: $DISPLAY_NAME"
echo "In Cursor, select the '$DISPLAY_NAME' kernel for notebooks in this repo."
