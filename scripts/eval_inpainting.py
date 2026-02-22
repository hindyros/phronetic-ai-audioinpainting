#!/usr/bin/env python3
"""Evaluate an audio inpainting model on test manifests."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an audio inpainting model.")
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Model checkpoint path."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/test.jsonl",
        help="Test manifest JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/eval",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for inference."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[eval_inpainting] args={args}")
    print("Evaluation not yet implemented. Complete later pipeline sections first.")
    sys.exit(0)


if __name__ == "__main__":
    main()
