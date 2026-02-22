#!/usr/bin/env python3
"""Train an audio inpainting model."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an audio inpainting model.")
    parser.add_argument("--config", type=str, default=None, help="Path to Hydra YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[train_inpainting] args={args}")
    print("Training not yet implemented. Complete later pipeline sections first.")
    sys.exit(0)


if __name__ == "__main__":
    main()
