#!/usr/bin/env python3
"""Preview telephony-augmented audio samples for sanity checking."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview telephony-style augmented audio samples."
    )
    parser.add_argument(
        "--input", type=str, required=False, help="Path to a clean audio file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/previews",
        help="Output directory for augmented samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of augmented variants to generate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/telephony.yaml",
        help="Telephony augmentation config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[preview_telephony_audio] args={args}")
    print("Telephony preview not yet implemented. Complete Section 2 first.")
    sys.exit(0)


if __name__ == "__main__":
    main()
