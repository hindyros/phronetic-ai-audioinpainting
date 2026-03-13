# Audio Inpainting

Hindi speech inpainting system for restoring corrupted or partially lost segments in recorded customer calls. This is a post-processing restoration pipeline (not real-time PLC) trained on the Vaani Hindi dataset.

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A HuggingFace account with access to [ARTPARK-IISc/Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd phronetic-ai-audioinpainting

# Install all dependencies (including dev tools)
uv sync --extra dev

# Set up pre-commit hooks
uv run pre-commit install

# Log in to HuggingFace (required for Vaani dataset)
uv run huggingface-cli login
```

### Notebook Setup In Cursor

Use the repo-local `.venv` so notebooks keep working when you reopen Cursor:

```bash
chmod +x scripts/setup_notebook_env.sh
./scripts/setup_notebook_env.sh
```

This creates or updates the shared environment at `~/.virtualenvs/global_notebooker`, installs the
project, notebook, and dev dependencies, and registers a stable Jupyter kernel named
`global_notebooker`.

Cursor will prefer `/Users/hindy/.virtualenvs/global_notebooker/bin/python` via `.vscode/settings.json`.
When you open a notebook, pick the `global_notebooker` kernel once and Cursor should keep reusing that project
environment across sessions.

## Project Structure

```
src/                  Core library code
  config.py           Pydantic configuration models
  utils/              Audio, STFT, distributed utilities
  data/               Dataset loaders and manifest tools
configs/              Hydra YAML configs
scripts/              CLI entry points
  prepare_vaani_hindi.py   Download and prepare Vaani Hindi data
  train_inpainting.py      Train inpainting model
  eval_inpainting.py       Evaluate model
  preview_telephony_audio.py  Preview telephony augmentation
  visualize_masks.py       Visualize packet-loss masks
data/
  raw/                Raw downloaded audio
  processed/          Preprocessed audio
  manifests/          JSONL train/val/test manifests
experiments/          Experiment outputs and checkpoints
tests/                Unit tests
docs/                 Documentation and reports
```

## Quick Start

### 1. Prepare data

```bash
# Download and prepare a small Hindi subset from Vaani
uv run python scripts/prepare_vaani_hindi.py \
    --districts MadhyaPradesh_Bhopal UttarPradesh_Lucknow \
    --output-dir data \
    --min-duration 1.0 \
    --max-duration 30.0
```

### 2. Train (coming soon)

```bash
uv run python scripts/train_inpainting.py --config configs/experiment.yaml
```

### 3. Evaluate (coming soon)

```bash
uv run python scripts/eval_inpainting.py --checkpoint experiments/best.ckpt
```

## CLI Help

All scripts support `--help`:

```bash
uv run python scripts/prepare_vaani_hindi.py --help
uv run python scripts/train_inpainting.py --help
uv run python scripts/eval_inpainting.py --help
uv run python scripts/preview_telephony_audio.py --help
uv run python scripts/visualize_masks.py --help
```

## Running Tests

```bash
uv run pytest
```

## Dataset

This project uses the **Vaani** dataset by ARTPARK and IISc Bangalore.

- **Source:** [ARTPARK-IISc/Vaani on HuggingFace](https://huggingface.co/datasets/ARTPARK-IISc/Vaani)
- **License:** CC-BY-4.0
- **Citation:**

```bibtex
@misc{vaani2025,
  author       = {VAANI Team},
  title        = {VAANI: Capturing the Language Landscape for an Inclusive Digital India},
  howpublished = {\url{https://vaani.iisc.ac.in/}},
  year         = {2025}
}
```

You must accept the dataset terms on HuggingFace and authenticate via `huggingface-cli login` before downloading.
