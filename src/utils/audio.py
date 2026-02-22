"""Audio I/O utilities: load, save, resample, normalize."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(
    path: str | Path,
    sr: int | None = 16_000,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and optionally resample / convert to mono.

    Returns (waveform, sample_rate) where waveform has shape (channels, samples).
    """
    waveform, orig_sr = torchaudio.load(str(path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr is not None and orig_sr != sr:
        waveform = resample(waveform, orig_sr, sr)
        orig_sr = sr

    return waveform, orig_sr


def save_audio(path: str | Path, waveform: torch.Tensor, sr: int) -> None:
    """Save a waveform tensor to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sr)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Peak-normalize a waveform so the absolute max equals target_peak."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform * (target_peak / peak)
    return waveform


def get_duration(path: str | Path) -> float:
    """Return duration in seconds without loading the full file."""
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate
