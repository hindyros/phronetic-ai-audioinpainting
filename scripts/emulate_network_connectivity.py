#!/usr/bin/env python3
"""Emulate mobile-network call issues on clean mono speech audio.

The primary corruption model inserts a configurable number of silent cuts
(packet-loss gaps) at random positions, each with a random duration drawn
uniformly from a caller-specified range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono if needed, and return float32 samples."""
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = waveform.mean(axis=1)
    return mono.astype(np.float32), sample_rate


def save_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Save a mono float waveform to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(waveform, -1.0, 1.0), sample_rate)


def ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    return max(1, int(round(milliseconds * sample_rate / 1000.0)))


# ---------------------------------------------------------------------------
# Core corruption: random silent cuts
# ---------------------------------------------------------------------------

def apply_random_cuts(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    num_cuts: int = 2,
    cut_ms_range: tuple[float, float] = (1.0, 200.0),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Insert *num_cuts* silent gaps at random positions.

    Each cut duration is drawn uniformly from *cut_ms_range* (in
    milliseconds).  Cut positions are drawn uniformly over the full
    waveform length, independently of each other (they may overlap,
    which is realistic for bursty packet loss).

    Parameters
    ----------
    audio:
        Float32 mono waveform.  Modified **in-place** and also returned.
    sample_rate:
        Sample rate of *audio*.
    rng:
        Numpy random generator for reproducibility.
    num_cuts:
        How many silent gaps to insert.
    cut_ms_range:
        ``(min_ms, max_ms)`` — uniform range for each cut's duration.

    Returns
    -------
    corrupted:
        The modified waveform (same object as *audio*, clipped to [-1, 1]).
    events:
        One dict per cut with timing metadata, sorted by start time.
    """
    total_samples = audio.shape[0]
    min_ms, max_ms = cut_ms_range
    events: list[dict[str, Any]] = []

    for _ in range(num_cuts):
        cut_ms = float(rng.uniform(min_ms, max_ms))
        cut_samples = ms_to_samples(cut_ms, sample_rate)
        cut_samples = min(cut_samples, total_samples)

        start = int(rng.integers(0, max(1, total_samples - cut_samples)))
        end = min(total_samples, start + cut_samples)

        audio[start:end] = 0.0

        events.append({
            "type": "cut",
            "start_sample": start,
            "end_sample": end,
            "start_sec": round(start / sample_rate, 4),
            "end_sec": round(end / sample_rate, 4),
            "duration_ms": round(cut_ms, 2),
        })

    events.sort(key=lambda e: e["start_sample"])
    return np.clip(audio, -1.0, 1.0), events


# ---------------------------------------------------------------------------
# High-level entry point used by notebooks and callers
# ---------------------------------------------------------------------------

def apply_network_artifacts(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    num_cuts: int = 2,
    cut_ms_range: tuple[float, float] = (1.0, 200.0),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply network-loss corruption to *audio* (convenience wrapper).

    Copies the input so the original array is never modified.
    """
    corrupted = audio.astype(np.float32, copy=True)
    corrupted, events = apply_random_cuts(
        corrupted, sample_rate, rng,
        num_cuts=num_cuts,
        cut_ms_range=cut_ms_range,
    )
    return corrupted, events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply random silent cuts to a clean WAV file to emulate network packet loss."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the clean WAV file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the corrupted WAV file.")
    parser.add_argument(
        "--num-cuts", type=int, default=2,
        help="Number of silent cuts to insert (default: 2).",
    )
    parser.add_argument(
        "--cut-min-ms", type=float, default=1.0,
        help="Minimum cut duration in milliseconds (default: 1).",
    )
    parser.add_argument(
        "--cut-max-ms", type=float, default=200.0,
        help="Maximum cut duration in milliseconds (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducible corruption patterns.",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Optional JSON report describing the sampled events.",
    )
    return parser.parse_args()


def build_report(
    args: argparse.Namespace,
    sample_rate: int,
    input_audio: np.ndarray,
    output_audio: np.ndarray,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a compact JSON-serializable report for pipeline logging."""
    return {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "sample_rate": sample_rate,
        "num_samples": int(input_audio.shape[0]),
        "duration_sec": round(input_audio.shape[0] / sample_rate, 4),
        "num_cuts": args.num_cuts,
        "cut_ms_range": [args.cut_min_ms, args.cut_max_ms],
        "seed": args.seed,
        "input_rms": round(float(np.sqrt(np.mean(np.square(input_audio)))), 6),
        "output_rms": round(float(np.sqrt(np.mean(np.square(output_audio)))), 6),
        "event_count": len(events),
        "events": events,
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    clean_audio, sample_rate = load_mono_audio(args.input)

    if sample_rate < 8_000 or sample_rate > 16_000:
        print(
            f"Warning: input sample rate is {sample_rate} Hz. "
            "This script is tuned for mono 8-16 kHz call audio."
        )

    corrupted_audio, events = apply_network_artifacts(
        clean_audio,
        sample_rate,
        rng=rng,
        num_cuts=args.num_cuts,
        cut_ms_range=(args.cut_min_ms, args.cut_max_ms),
    )
    save_audio(args.output, corrupted_audio, sample_rate)

    if args.report is not None:
        report = build_report(args, sample_rate, clean_audio, corrupted_audio, events)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))

    print(
        f"Wrote {args.output} with {len(events)} cut(s), "
        f"duration range [{args.cut_min_ms}-{args.cut_max_ms}] ms, seed={args.seed}."
    )


if __name__ == "__main__":
    main()
