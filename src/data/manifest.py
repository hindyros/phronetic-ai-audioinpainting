"""Manifest generation: speaker-based splits and JSONL I/O."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"audio_path", "speaker_id", "duration", "language", "split"}


def create_speaker_splits(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Assign each record to train / val / test based on speaker_id.

    All utterances from a single speaker go into the same split to avoid
    speaker leakage across splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Group by speaker
    speaker_to_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        speaker_to_records[r["speaker_id"]].append(r)

    speakers = sorted(speaker_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_spk = set(speakers[:n_train])
    val_spk = set(speakers[n_train : n_train + n_val])
    test_spk = set(speakers[n_train + n_val :])

    # Handle edge case: if only 1-2 speakers, ensure at least train exists
    if not test_spk and len(speakers) > 1:
        test_spk = val_spk
        val_spk = set()

    out: list[dict[str, Any]] = []
    for spk, recs in speaker_to_records.items():
        if spk in train_spk:
            split = "train"
        elif spk in val_spk:
            split = "val"
        else:
            split = "test"
        for r in recs:
            out.append({**r, "split": split})

    logger.info(
        "Speaker splits: %d train, %d val, %d test speakers (%d total records)",
        len(train_spk),
        len(val_spk),
        len(test_spk),
        len(out),
    )
    return out


def write_manifest(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write records to a JSONL manifest file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), output_path)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest file back into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_split_manifests(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write separate JSONL files per split (train.jsonl, val.jsonl, test.jsonl)."""
    output_dir = Path(output_dir)
    splits: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        splits[r["split"]].append(r)

    paths: dict[str, Path] = {}
    for split_name, split_records in splits.items():
        p = output_dir / f"{split_name}.jsonl"
        write_manifest(split_records, p)
        paths[split_name] = p

    return paths


def validate_manifest(records: list[dict[str, Any]]) -> list[str]:
    """Return a list of validation errors (empty list means valid)."""
    errors = []
    for i, r in enumerate(records):
        missing = REQUIRED_FIELDS - set(r.keys())
        if missing:
            errors.append(f"Row {i}: missing fields {missing}")
        if r.get("language") != "hi":
            errors.append(f"Row {i}: language is '{r.get('language')}', expected 'hi'")
    return errors
