"""Lightweight wrappers for distributed training with PyTorch."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize the default distributed process group.

    Expects RANK, WORLD_SIZE, and MASTER_ADDR / MASTER_PORT environment
    variables to be set (standard for torchrun / Lightning).
    Falls back to gloo on CPU-only machines.
    """
    if dist.is_initialized():
        return

    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"

    dist.init_process_group(backend=backend)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()
