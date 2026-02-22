"""STFT / iSTFT utilities and multi-resolution STFT loss."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as f


def stft(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
) -> torch.Tensor:
    """Compute the complex STFT of a waveform.

    Args:
        waveform: (batch, samples) or (batch, 1, samples)
        n_fft: FFT size
        hop_length: hop between frames
        win_length: window length (defaults to n_fft)

    Returns:
        Complex tensor of shape (batch, freq_bins, time_frames).
    """
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)

    win_length = win_length or n_fft
    window = torch.hann_window(win_length, device=waveform.device)

    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )


def istft(
    spec: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int | None = None,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse STFT back to waveform.

    Args:
        spec: Complex tensor (batch, freq_bins, time_frames)
        length: desired output length in samples

    Returns:
        Waveform tensor (batch, samples).
    """
    win_length = win_length or n_fft
    window = torch.hann_window(win_length, device=spec.device)

    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def _spectral_convergence(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    return torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-7)


def _log_magnitude_loss(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
    return f.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss combining spectral convergence and log-magnitude L1."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (128, 256, 512),
        win_sizes: tuple[int, ...] = (512, 1024, 2048),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-resolution STFT loss.

        Args:
            predicted: (batch, samples) predicted waveform
            target:    (batch, samples) ground-truth waveform

        Returns:
            Scalar loss tensor.
        """
        if predicted.dim() == 3:
            predicted = predicted.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        loss = torch.tensor(0.0, device=predicted.device)

        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_spec = stft(predicted, n_fft=n_fft, hop_length=hop, win_length=win)
            tgt_spec = stft(target, n_fft=n_fft, hop_length=hop, win_length=win)

            pred_mag = pred_spec.abs()
            tgt_mag = tgt_spec.abs()

            loss = loss + _spectral_convergence(pred_mag, tgt_mag)
            loss = loss + _log_magnitude_loss(pred_mag, tgt_mag)

        return loss / len(self.fft_sizes)
