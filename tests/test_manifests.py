"""Unit tests for manifest generation and validation."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from src.data.manifest import (
    REQUIRED_FIELDS,
    create_speaker_splits,
    read_manifest,
    validate_manifest,
    write_manifest,
    write_split_manifests,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_records(n_speakers: int = 10, utts_per_speaker: int = 5) -> list[dict]:
    """Generate synthetic manifest records for testing."""
    records = []
    for s in range(n_speakers):
        for u in range(utts_per_speaker):
            records.append(
                {
                    "audio_path": f"/tmp/audio/spk{s:03d}_{u:03d}.wav",
                    "speaker_id": f"spk{s:03d}",
                    "duration": 3.5 + u * 0.1,
                    "language": "hi",
                }
            )
    return records


@pytest.fixture
def sample_records():
    return _make_records()


@pytest.fixture
def split_records(sample_records):
    return create_speaker_splits(sample_records, seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpeakerSplits:
    def test_no_speaker_overlap(self, split_records):
        """Train, val, and test must have disjoint speaker sets."""
        splits: dict[str, set] = defaultdict(set)
        for r in split_records:
            splits[r["split"]].add(r["speaker_id"])

        train = splits.get("train", set())
        val = splits.get("val", set())
        test = splits.get("test", set())

        assert train.isdisjoint(val), f"Train/val overlap: {train & val}"
        assert train.isdisjoint(test), f"Train/test overlap: {train & test}"
        assert val.isdisjoint(test), f"Val/test overlap: {val & test}"

    def test_all_records_assigned(self, sample_records, split_records):
        """Every input record should appear in exactly one split."""
        assert len(split_records) == len(sample_records)
        for r in split_records:
            assert r["split"] in {"train", "val", "test"}

    def test_deterministic(self, sample_records):
        """Same seed should produce identical splits."""
        a = create_speaker_splits(sample_records, seed=123)
        b = create_speaker_splits(sample_records, seed=123)
        for ra, rb in zip(a, b):
            assert ra["split"] == rb["split"]
            assert ra["speaker_id"] == rb["speaker_id"]


class TestLanguageField:
    def test_all_language_hi(self, split_records):
        """All records must have language == 'hi'."""
        for r in split_records:
            assert r["language"] == "hi", f"Expected 'hi', got '{r['language']}'"


class TestManifestSchema:
    def test_required_fields_present(self, split_records):
        """Every record must contain all required fields."""
        for i, r in enumerate(split_records):
            missing = REQUIRED_FIELDS - set(r.keys())
            assert not missing, f"Row {i} missing: {missing}"


class TestManifestIO:
    def test_roundtrip(self, split_records):
        """Write then read should produce identical records."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        write_manifest(split_records, path)
        loaded = read_manifest(path)

        assert len(loaded) == len(split_records)
        for orig, loaded_r in zip(split_records, loaded):
            assert orig == loaded_r

        path.unlink()

    def test_write_split_manifests(self, split_records):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_split_manifests(split_records, tmpdir)
            assert "train" in paths

            for split_name, p in paths.items():
                records = read_manifest(p)
                assert all(r["split"] == split_name for r in records)


class TestValidation:
    def test_valid_records_pass(self, split_records):
        errors = validate_manifest(split_records)
        assert errors == []

    def test_missing_field_detected(self):
        bad = [{"audio_path": "/x.wav", "speaker_id": "s", "duration": 1.0, "language": "hi"}]
        errors = validate_manifest(bad)
        assert any("missing" in e for e in errors)

    def test_wrong_language_detected(self):
        bad = [
            {
                "audio_path": "/x.wav",
                "speaker_id": "s",
                "duration": 1.0,
                "language": "en",
                "split": "train",
            }
        ]
        errors = validate_manifest(bad)
        assert any("language" in e for e in errors)


class TestFilesReadable:
    """Spot-check that audio paths in manifests exist.

    Skipped if data hasn't been downloaded yet.
    """

    @pytest.mark.skipif(
        not Path("data/manifests/train.jsonl").exists(),
        reason="Manifests not generated yet (run prepare_vaani_hindi.py first)",
    )
    def test_files_exist(self):
        records = read_manifest("data/manifests/train.jsonl")
        # Check up to 5 files
        for r in records[:5]:
            p = Path(r["audio_path"])
            assert p.exists(), f"Audio file not found: {p}"
            assert p.stat().st_size > 0, f"Audio file is empty: {p}"
