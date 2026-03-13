#!/usr/bin/env python3
"""Emulate mobile-network call issues on clean mono speech audio."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class SeverityProfile:
    """Configuration bundle controlling event frequency and strength."""

    # Packet-loss-like cuts: very short speech dropouts, sometimes clustered.
    short_dropouts_per_minute: tuple[int, int]
    short_dropout_ms: tuple[int, int]
    short_cluster_probability: float
    short_cluster_size: tuple[int, int]
    short_gap_ms: tuple[int, int]
    short_noise_probability: float
    short_noise_level: float
    # Temporary network loss or tower handover: longer masked outages.
    long_cuts_per_minute: tuple[float, float]
    long_cut_ms: tuple[int, int]
    long_noise_probability: float
    long_noise_level: float
    # Congestion: rapid alternation between audible and missing speech.
    choppy_windows_per_minute: tuple[float, float]
    choppy_window_ms: tuple[int, int]
    choppy_on_ms: tuple[int, int]
    choppy_off_ms: tuple[int, int]
    choppy_noise_level: float
    # Weak signal/interference: fading, noise, and occasional dull band-limiting.
    degradation_windows_per_minute: tuple[float, float]
    degradation_window_ms: tuple[int, int]
    fade_segment_ms: tuple[int, int]
    max_fade_attenuation_db: float
    broadband_noise_std: float
    bandlimit_probability: float
    bandlimit_cutoff_hz: tuple[int, int]
    static_burst_probability: float
    static_burst_ms: tuple[int, int]
    static_burst_level: float


SEVERITY_PROFILES: dict[str, SeverityProfile] = {
    "light": SeverityProfile(
        short_dropouts_per_minute=(2, 5),
        short_dropout_ms=(20, 90),
        short_cluster_probability=0.20,
        short_cluster_size=(2, 3),
        short_gap_ms=(10, 40),
        short_noise_probability=0.35,
        short_noise_level=0.003,
        long_cuts_per_minute=(0.1, 0.4),
        long_cut_ms=(500, 1400),
        long_noise_probability=0.30,
        long_noise_level=0.0015,
        choppy_windows_per_minute=(0.2, 0.8),
        choppy_window_ms=(1000, 2500),
        choppy_on_ms=(80, 150),
        choppy_off_ms=(60, 130),
        choppy_noise_level=0.002,
        degradation_windows_per_minute=(0.6, 1.2),
        degradation_window_ms=(1200, 2600),
        fade_segment_ms=(200, 500),
        max_fade_attenuation_db=7.0,
        broadband_noise_std=0.0025,
        bandlimit_probability=0.20,
        bandlimit_cutoff_hz=(2000, 3200),
        static_burst_probability=0.20,
        static_burst_ms=(8, 20),
        static_burst_level=0.010,
    ),
    "medium": SeverityProfile(
        short_dropouts_per_minute=(4, 10),
        short_dropout_ms=(30, 140),
        short_cluster_probability=0.35,
        short_cluster_size=(2, 4),
        short_gap_ms=(10, 50),
        short_noise_probability=0.45,
        short_noise_level=0.005,
        long_cuts_per_minute=(0.3, 0.8),
        long_cut_ms=(700, 2200),
        long_noise_probability=0.40,
        long_noise_level=0.002,
        choppy_windows_per_minute=(0.5, 1.3),
        choppy_window_ms=(1500, 3500),
        choppy_on_ms=(60, 130),
        choppy_off_ms=(60, 150),
        choppy_noise_level=0.003,
        degradation_windows_per_minute=(1.0, 1.8),
        degradation_window_ms=(1500, 3200),
        fade_segment_ms=(150, 400),
        max_fade_attenuation_db=12.0,
        broadband_noise_std=0.0045,
        bandlimit_probability=0.40,
        bandlimit_cutoff_hz=(1600, 2800),
        static_burst_probability=0.35,
        static_burst_ms=(10, 28),
        static_burst_level=0.016,
    ),
    "heavy": SeverityProfile(
        short_dropouts_per_minute=(8, 16),
        short_dropout_ms=(40, 200),
        short_cluster_probability=0.55,
        short_cluster_size=(2, 5),
        short_gap_ms=(5, 60),
        short_noise_probability=0.55,
        short_noise_level=0.008,
        long_cuts_per_minute=(0.6, 1.4),
        long_cut_ms=(1000, 3000),
        long_noise_probability=0.50,
        long_noise_level=0.003,
        choppy_windows_per_minute=(0.9, 2.0),
        choppy_window_ms=(2000, 5000),
        choppy_on_ms=(50, 120),
        choppy_off_ms=(50, 150),
        choppy_noise_level=0.004,
        degradation_windows_per_minute=(1.4, 2.5),
        degradation_window_ms=(1800, 4500),
        fade_segment_ms=(120, 320),
        max_fade_attenuation_db=18.0,
        broadband_noise_std=0.007,
        bandlimit_probability=0.60,
        bandlimit_cutoff_hz=(1200, 2400),
        static_burst_probability=0.50,
        static_burst_ms=(12, 35),
        static_burst_level=0.022,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply mobile-network connectivity artifacts to a clean WAV file."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the clean WAV file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the corrupted WAV file.")
    parser.add_argument(
        "--severity",
        type=str,
        choices=sorted(SEVERITY_PROFILES),
        default="medium",
        help="Overall corruption severity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible corruption patterns.",
    )
    parser.add_argument(
        "--effects",
        nargs="+",
        choices=["short_dropouts", "long_cuts", "choppy", "degradation", "all"],
        default=["all"],
        help="Subset of effects to apply.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report describing the sampled events.",
    )
    return parser.parse_args()


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


def choose_count(
    duration_seconds: float,
    per_minute_range: tuple[int | float, int | float],
    rng: np.random.Generator,
) -> int:
    """Draw an event count from a duration-scaled uniform rate."""
    low_rate, high_rate = per_minute_range
    sampled_rate = rng.uniform(low_rate, high_rate)
    expected = sampled_rate * duration_seconds / 60.0
    return int(rng.poisson(expected))


def sample_interval(
    total_samples: int,
    length_samples: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Pick a valid start/end interval inside a waveform."""
    if length_samples >= total_samples:
        return 0, total_samples
    start = int(rng.integers(0, total_samples - length_samples))
    return start, start + length_samples


def add_static_burst(
    audio: np.ndarray,
    sample_rate: int,
    center_sample: int,
    rng: np.random.Generator,
    profile: SeverityProfile,
) -> dict[str, Any]:
    """Inject a brief crackle burst around a transition point."""
    burst_samples = ms_to_samples(
        rng.uniform(*profile.static_burst_ms), sample_rate
    )
    start = max(0, center_sample - burst_samples // 2)
    end = min(audio.shape[0], start + burst_samples)
    if end <= start:
        return {}

    burst = rng.normal(0.0, profile.static_burst_level, end - start)
    spike_mask = rng.random(end - start) < 0.12
    burst[spike_mask] += rng.normal(0.0, profile.static_burst_level * 2.5, spike_mask.sum())
    audio[start:end] += burst.astype(np.float32)

    return {
        "type": "static_burst",
        "start_sample": start,
        "end_sample": end,
        "start_sec": round(start / sample_rate, 4),
        "end_sec": round(end / sample_rate, 4),
    }


def lowpass_fft(segment: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Apply a simple FFT-domain low-pass filter."""
    if segment.size < 8:
        return segment

    spectrum = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    transition_hz = max(100.0, cutoff_hz * 0.15)
    lower = max(0.0, cutoff_hz - transition_hz)
    upper = min(sample_rate / 2.0, cutoff_hz + transition_hz)

    mask = np.ones_like(freqs)
    mask[freqs >= upper] = 0.0
    transition = (freqs > lower) & (freqs < upper)
    if np.any(transition):
        phase = (freqs[transition] - lower) / max(upper - lower, 1.0)
        mask[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))

    filtered = np.fft.irfft(spectrum * mask, n=segment.size)
    return filtered.astype(np.float32)


def apply_short_dropouts(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    profile: SeverityProfile,
    force_count: int | None = None,
) -> list[dict[str, Any]]:
    """Insert short packet-loss-like mutes or noisy dropouts."""
    events: list[dict[str, Any]] = []
    total_samples = audio.shape[0]
    duration_seconds = total_samples / sample_rate
    clusters = force_count or choose_count(duration_seconds, profile.short_dropouts_per_minute, rng)

    for _ in range(clusters):
        cluster_size = 1
        if rng.random() < profile.short_cluster_probability:
            cluster_size = int(
                rng.integers(
                    profile.short_cluster_size[0],
                    profile.short_cluster_size[1] + 1,
                )
            )

        base_length = ms_to_samples(
            rng.uniform(*profile.short_dropout_ms),
            sample_rate,
        )
        start, _ = sample_interval(total_samples, base_length, rng)

        cursor = start
        for _ in range(cluster_size):
            dropout_samples = ms_to_samples(
                rng.uniform(*profile.short_dropout_ms),
                sample_rate,
            )
            gap_samples = ms_to_samples(
                rng.uniform(*profile.short_gap_ms),
                sample_rate,
            )

            end = min(total_samples, cursor + dropout_samples)
            if end <= cursor:
                continue

            replacement = np.zeros(end - cursor, dtype=np.float32)
            dropout_kind = "silence"
            if rng.random() < profile.short_noise_probability:
                dropout_kind = "noise_floor"
                replacement = rng.normal(
                    0.0,
                    profile.short_noise_level,
                    end - cursor,
                ).astype(np.float32)

            audio[cursor:end] = replacement
            event = {
                "type": "short_dropout",
                "variant": dropout_kind,
                "start_sample": cursor,
                "end_sample": end,
                "start_sec": round(cursor / sample_rate, 4),
                "end_sec": round(end / sample_rate, 4),
            }
            events.append(event)

            if rng.random() < profile.static_burst_probability:
                burst_event = add_static_burst(audio, sample_rate, cursor, rng, profile)
                if burst_event:
                    events.append(burst_event)
            if rng.random() < profile.static_burst_probability:
                burst_event = add_static_burst(audio, sample_rate, end, rng, profile)
                if burst_event:
                    events.append(burst_event)

            cursor = min(total_samples, end + gap_samples)

    return events


def apply_long_connection_cuts(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    profile: SeverityProfile,
    force_count: int | None = None,
) -> list[dict[str, Any]]:
    """Mask longer sections to emulate temporary network loss or handover."""
    events: list[dict[str, Any]] = []
    total_samples = audio.shape[0]
    duration_seconds = total_samples / sample_rate
    cut_count = force_count or choose_count(duration_seconds, profile.long_cuts_per_minute, rng)

    for _ in range(cut_count):
        cut_samples = ms_to_samples(rng.uniform(*profile.long_cut_ms), sample_rate)
        start, end = sample_interval(total_samples, cut_samples, rng)

        replacement = np.zeros(end - start, dtype=np.float32)
        cut_kind = "silence"
        if rng.random() < profile.long_noise_probability:
            cut_kind = "noisy_open_line"
            replacement = rng.normal(
                0.0,
                profile.long_noise_level,
                end - start,
            ).astype(np.float32)

        audio[start:end] = replacement
        events.append(
            {
                "type": "long_cut",
                "variant": cut_kind,
                "start_sample": start,
                "end_sample": end,
                "start_sec": round(start / sample_rate, 4),
                "end_sec": round(end / sample_rate, 4),
            }
        )

        if rng.random() < profile.static_burst_probability:
            burst_event = add_static_burst(audio, sample_rate, start, rng, profile)
            if burst_event:
                events.append(burst_event)
        if rng.random() < profile.static_burst_probability:
            burst_event = add_static_burst(audio, sample_rate, end, rng, profile)
            if burst_event:
                events.append(burst_event)

    return events


def apply_choppy_congestion(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    profile: SeverityProfile,
    force_count: int | None = None,
) -> list[dict[str, Any]]:
    """Alternate short on/off regions to mimic congested network transmission."""
    events: list[dict[str, Any]] = []
    total_samples = audio.shape[0]
    duration_seconds = total_samples / sample_rate
    window_count = force_count or choose_count(duration_seconds, profile.choppy_windows_per_minute, rng)

    for _ in range(window_count):
        window_samples = ms_to_samples(
            rng.uniform(*profile.choppy_window_ms),
            sample_rate,
        )
        start, end = sample_interval(total_samples, window_samples, rng)
        cursor = start
        phase = "pass"

        while cursor < end:
            if phase == "pass":
                span = ms_to_samples(rng.uniform(*profile.choppy_on_ms), sample_rate)
                seg_end = min(end, cursor + span)
                attenuation_db = rng.uniform(0.5, 3.0)
                gain = 10 ** (-attenuation_db / 20.0)
                audio[cursor:seg_end] *= gain
                phase = "drop"
            else:
                span = ms_to_samples(rng.uniform(*profile.choppy_off_ms), sample_rate)
                seg_end = min(end, cursor + span)
                replacement = rng.normal(
                    0.0,
                    profile.choppy_noise_level,
                    seg_end - cursor,
                ).astype(np.float32)
                audio[cursor:seg_end] = replacement
                phase = "pass"
            cursor = seg_end

        events.append(
            {
                "type": "choppy_window",
                "start_sample": start,
                "end_sample": end,
                "start_sec": round(start / sample_rate, 4),
                "end_sec": round(end / sample_rate, 4),
            }
        )

        if rng.random() < profile.static_burst_probability:
            burst_event = add_static_burst(audio, sample_rate, start, rng, profile)
            if burst_event:
                events.append(burst_event)
        if rng.random() < profile.static_burst_probability:
            burst_event = add_static_burst(audio, sample_rate, end, rng, profile)
            if burst_event:
                events.append(burst_event)

    return events


def apply_signal_degradation(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    profile: SeverityProfile,
    force_count: int | None = None,
) -> list[dict[str, Any]]:
    """Apply fading, broadband noise, and occasional band-limiting."""
    events: list[dict[str, Any]] = []
    total_samples = audio.shape[0]
    duration_seconds = total_samples / sample_rate
    window_count = force_count or choose_count(
        duration_seconds,
        profile.degradation_windows_per_minute,
        rng,
    )

    for _ in range(window_count):
        window_samples = ms_to_samples(
            rng.uniform(*profile.degradation_window_ms),
            sample_rate,
        )
        start, end = sample_interval(total_samples, window_samples, rng)
        if end <= start:
            continue

        segment = audio[start:end].copy()
        anchor_stride = ms_to_samples(rng.uniform(*profile.fade_segment_ms), sample_rate)
        anchor_positions = np.arange(0, segment.size, anchor_stride, dtype=int)
        if anchor_positions.size == 0 or anchor_positions[-1] != segment.size - 1:
            anchor_positions = np.append(anchor_positions, segment.size - 1)

        attenuation_db = rng.uniform(
            0.0,
            profile.max_fade_attenuation_db,
            size=anchor_positions.shape[0],
        )
        anchor_gains = 10 ** (-attenuation_db / 20.0)
        envelope = np.interp(np.arange(segment.size), anchor_positions, anchor_gains).astype(
            np.float32
        )

        segment *= envelope
        segment += rng.normal(0.0, profile.broadband_noise_std, segment.size).astype(np.float32)

        bandlimited = False
        cutoff_hz = None
        if rng.random() < profile.bandlimit_probability:
            cutoff_hz = float(rng.uniform(*profile.bandlimit_cutoff_hz))
            segment = lowpass_fft(segment, sample_rate, cutoff_hz)
            bandlimited = True

        audio[start:end] = segment
        event = {
            "type": "signal_degradation",
            "bandlimited": bandlimited,
            "start_sample": start,
            "end_sample": end,
            "start_sec": round(start / sample_rate, 4),
            "end_sec": round(end / sample_rate, 4),
            "max_attenuation_db": round(float(attenuation_db.max()), 2),
        }
        if cutoff_hz is not None:
            event["cutoff_hz"] = round(cutoff_hz, 1)
        events.append(event)

    return events


def apply_network_artifacts(
    audio: np.ndarray,
    sample_rate: int,
    severity: str,
    rng: np.random.Generator,
    effects: set[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply the requested network effects while preserving audio length."""
    profile = SEVERITY_PROFILES[severity]
    enabled_effects = effects or {
        "short_dropouts",
        "long_cuts",
        "choppy",
        "degradation",
    }

    corrupted = audio.astype(np.float32, copy=True)
    events: list[dict[str, Any]] = []

    if "degradation" in enabled_effects:
        # Weak coverage makes speech fade, get noisy, and lose high-frequency detail.
        events.extend(apply_signal_degradation(corrupted, sample_rate, rng, profile))
    if "short_dropouts" in enabled_effects:
        # Lost or delayed packets create brief holes in otherwise continuous speech.
        events.extend(apply_short_dropouts(corrupted, sample_rate, rng, profile))
    if "choppy" in enabled_effects:
        # Congestion makes the call alternate between audible and missing slices.
        events.extend(apply_choppy_congestion(corrupted, sample_rate, rng, profile))
    if "long_cuts" in enabled_effects:
        # A temporary disconnect masks a longer region without changing alignment.
        events.extend(apply_long_connection_cuts(corrupted, sample_rate, rng, profile))

    if not events and corrupted.shape[0] >= ms_to_samples(120, sample_rate):
        fallback_order = [
            "short_dropouts",
            "degradation",
            "choppy",
            "long_cuts",
        ]
        fallback_effect = next(
            (effect_name for effect_name in fallback_order if effect_name in enabled_effects),
            None,
        )
        if fallback_effect == "short_dropouts":
            events.extend(
                apply_short_dropouts(
                    corrupted,
                    sample_rate,
                    rng,
                    profile,
                    force_count=1,
                )
            )
        elif fallback_effect == "degradation":
            events.extend(
                apply_signal_degradation(
                    corrupted,
                    sample_rate,
                    rng,
                    profile,
                    force_count=1,
                )
            )
        elif fallback_effect == "choppy":
            events.extend(
                apply_choppy_congestion(
                    corrupted,
                    sample_rate,
                    rng,
                    profile,
                    force_count=1,
                )
            )
        elif fallback_effect == "long_cuts":
            events.extend(
                apply_long_connection_cuts(
                    corrupted,
                    sample_rate,
                    rng,
                    profile,
                    force_count=1,
                )
            )

    return np.clip(corrupted, -1.0, 1.0), sorted(events, key=lambda item: item["start_sample"])


def resolve_effects(selected_effects: list[str]) -> set[str]:
    if "all" in selected_effects:
        return {"short_dropouts", "long_cuts", "choppy", "degradation"}
    return set(selected_effects)


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
        "severity": args.severity,
        "seed": args.seed,
        "effects": sorted(resolve_effects(args.effects)),
        "profile": asdict(SEVERITY_PROFILES[args.severity]),
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

    effects = resolve_effects(args.effects)
    corrupted_audio, events = apply_network_artifacts(
        clean_audio,
        sample_rate,
        severity=args.severity,
        rng=rng,
        effects=effects,
    )
    save_audio(args.output, corrupted_audio, sample_rate)

    if args.report is not None:
        report = build_report(args, sample_rate, clean_audio, corrupted_audio, events)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))

    print(
        f"Wrote {args.output} with {len(events)} event(s) "
        f"at severity='{args.severity}' and seed={args.seed}."
    )


if __name__ == "__main__":
    main()
