"""Vaani Hindi dataset loading, filtering, and audio export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Dataset, get_dataset_config_names, load_dataset

logger = logging.getLogger(__name__)

DATASET_ID = "ARTPARK-IISc/VAANI"

# Districts with predominantly Hindi audio (good for a small dev subset).
DEFAULT_DISTRICTS = [
    "MadhyaPradesh_Bhopal",
    "UttarPradesh_Lucknow",
]

# The Vaani dataset labels Hindi in several ways across districts.
_HINDI_LABELS = {"Hindi", "hindi", "hi"}


def list_available_subsets(token: str | None = None) -> list[str]:
    """Return all available {State}_{District} subset names."""
    return get_dataset_config_names(DATASET_ID, token=token)


def load_district(subset: str, token: str | None = None) -> Dataset:
    """Load a single district subset from HuggingFace."""
    logger.info("Loading subset %s from %s ...", subset, DATASET_ID)
    ds = load_dataset(DATASET_ID, subset, token=token, trust_remote_code=True)
    if isinstance(ds, dict):
        # Prefer the "train" split if present; otherwise take the first.
        ds = ds.get("train", next(iter(ds.values())))
    return ds


def _is_hindi(example: dict[str, Any], language_column: str) -> bool:
    lang = example.get(language_column, "")
    if isinstance(lang, str):
        return lang.strip() in _HINDI_LABELS
    return False


def detect_language_column(ds: Dataset) -> str:
    """Heuristically find the column that contains the language label."""
    candidates = ["language", "lang", "Language", "Lang"]
    for c in candidates:
        if c in ds.column_names:
            return c
    # Fall back: look for any column whose values include "Hindi"
    for col in ds.column_names:
        sample = ds[0].get(col)
        if isinstance(sample, str) and sample.strip() in _HINDI_LABELS:
            return col
    raise ValueError(
        f"Cannot detect language column. Available columns: {ds.column_names}"
    )


def detect_speaker_column(ds: Dataset) -> str | None:
    """Heuristically find the column that contains speaker IDs."""
    candidates = ["speaker_id", "speakerid", "speaker", "Speaker_ID"]
    for c in candidates:
        if c in ds.column_names:
            return c
    return None


def detect_audio_column(ds: Dataset) -> str:
    """Find the column containing audio data."""
    candidates = ["audio", "Audio"]
    for c in candidates:
        if c in ds.column_names:
            return c
    raise ValueError(f"Cannot detect audio column. Available columns: {ds.column_names}")


def filter_hindi(ds: Dataset, language_column: str | None = None) -> Dataset:
    """Keep only rows where the language is Hindi."""
    if language_column is None:
        language_column = detect_language_column(ds)
    before = len(ds)
    ds = ds.filter(lambda ex: _is_hindi(ex, language_column))
    logger.info("Hindi filter: %d -> %d rows", before, len(ds))
    return ds


def filter_duration(
    ds: Dataset,
    min_dur: float = 1.0,
    max_dur: float = 30.0,
    audio_column: str = "audio",
) -> Dataset:
    """Keep rows with duration within [min_dur, max_dur] seconds."""

    def _in_range(example: dict) -> bool:
        audio = example.get(audio_column)
        if audio is None:
            return False
        # HF audio feature stores {"array": ..., "sampling_rate": ...}
        if isinstance(audio, dict):
            arr = audio.get("array")
            sr = audio.get("sampling_rate", 16_000)
            if arr is not None:
                dur = len(arr) / sr
                return min_dur <= dur <= max_dur
        return True  # keep if we can't compute duration

    before = len(ds)
    ds = ds.filter(_in_range)
    logger.info("Duration filter [%.1f, %.1f]s: %d -> %d rows", min_dur, max_dur, before, len(ds))
    return ds


def export_audio(
    ds: Dataset,
    output_dir: Path,
    audio_column: str = "audio",
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Save each row's audio to disk and return a list of metadata records.

    Each record contains: audio_path, speaker_id, duration, language.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_col = detect_speaker_column(ds)
    audio_col = audio_column if audio_column in ds.column_names else detect_audio_column(ds)

    records: list[dict[str, Any]] = []

    for idx, example in enumerate(ds):
        audio = example[audio_col]
        if not isinstance(audio, dict) or "array" not in audio:
            logger.warning("Skipping row %d: no audio array", idx)
            continue

        array = audio["array"]
        sr = audio.get("sampling_rate", 16_000)
        duration = len(array) / sr

        fname = f"{prefix}{idx:06d}.wav"
        audio_path = output_dir / fname
        sf.write(str(audio_path), array, sr)

        records.append(
            {
                "audio_path": str(audio_path),
                "speaker_id": (
                    example.get(speaker_col, f"spk_{idx}") if speaker_col else f"spk_{idx}"
                ),
                "duration": round(duration, 4),
                "language": "hi",
            }
        )

    logger.info("Exported %d audio files to %s", len(records), output_dir)
    return records
