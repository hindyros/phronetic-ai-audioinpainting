"""Pydantic-based configuration models for the audio inpainting pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


def get_git_commit_hash() -> str | None:
    """Return the current short git commit hash, or None if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class AudioConfig(BaseModel):
    """Audio processing parameters."""

    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    mono: bool = True


class DataConfig(BaseModel):
    """Dataset and manifest configuration.

    Paths are language-specific. Use ``data/{raw,processed,manifests}/english``
    for the English inpainting model and ``data/{raw,processed,manifests}/hindi``
    for the Hindi Vaani dataset.
    """

    language: str = "en"
    manifest_dir: Path = Path("data/manifests/english")
    raw_dir: Path = Path("data/raw/english")
    processed_dir: Path = Path("data/processed/english")
    min_duration: float = 1.0
    max_duration: float = 30.0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


class TrainingConfig(BaseModel):
    """Training hyper-parameters."""

    lr: float = 3e-4
    batch_size: int = 16
    epochs: int = 100
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    num_workers: int = 4
    precision: str = "bf16-mixed"
    accelerator: str = "auto"


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    project: str = "audio-inpainting"
    entity: str | None = None
    enabled: bool = True
    log_every_n_steps: int = 50


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration that composes all sub-configs."""

    name: str = "default"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    git_hash: str | None = Field(default_factory=get_git_commit_hash)

    def save(self, path: Path) -> None:
        """Persist the full config as JSON alongside an experiment run."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.model_validate_json(Path(path).read_text())
