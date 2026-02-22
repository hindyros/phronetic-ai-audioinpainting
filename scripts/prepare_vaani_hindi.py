#!/usr/bin/env python3
"""Download, filter, and prepare Vaani Hindi data into JSONL manifests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root: `python scripts/prepare_vaani_hindi.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import create_speaker_splits, validate_manifest, write_split_manifests
from src.data.vaani import (
    DEFAULT_DISTRICTS,
    export_audio,
    filter_duration,
    filter_hindi,
    list_available_subsets,
    load_district,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare Hindi audio from the Vaani dataset."
    )
    parser.add_argument(
        "--districts",
        nargs="+",
        default=None,
        help=(
            "List of {State}_{District} subset names to download. "
            "Defaults to a small curated list. "
            "Pass 'list' as the first argument to print all available subsets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Root output directory (default: data/).",
    )
    parser.add_argument(
        "--min-duration", type=float, default=1.0, help="Min utterance duration (s)."
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0, help="Max utterance duration (s)."
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    parser.add_argument(
        "--token", type=str, default=None, help="HuggingFace access token."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Handle the special "list" command
    if args.districts and args.districts[0].lower() == "list":
        subsets = list_available_subsets(token=args.token)
        print(f"\n{len(subsets)} available subsets:\n")
        for s in sorted(subsets):
            print(f"  {s}")
        return

    districts = args.districts or DEFAULT_DISTRICTS
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    manifest_dir = output_dir / "manifests"

    all_records: list[dict] = []

    for district in districts:
        logger.info("=== Processing district: %s ===", district)
        try:
            ds = load_district(district, token=args.token)
        except Exception:
            logger.exception("Failed to load district %s, skipping.", district)
            continue

        logger.info("Loaded %d rows, columns: %s", len(ds), ds.column_names)

        ds = filter_hindi(ds)
        if len(ds) == 0:
            logger.warning("No Hindi rows found in %s, skipping.", district)
            continue

        ds = filter_duration(ds, min_dur=args.min_duration, max_dur=args.max_duration)
        if len(ds) == 0:
            logger.warning("No rows in duration range for %s, skipping.", district)
            continue

        records = export_audio(ds, raw_dir / district, prefix=f"{district}_")
        all_records.extend(records)

    if not all_records:
        logger.error("No records produced. Check district names and filters.")
        sys.exit(1)

    logger.info("Total records across all districts: %d", len(all_records))

    # Create speaker-based splits
    all_records = create_speaker_splits(
        all_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Validate
    errors = validate_manifest(all_records)
    if errors:
        for e in errors[:20]:
            logger.warning("Validation: %s", e)
        logger.warning("Total validation errors: %d", len(errors))

    # Write split manifests
    paths = write_split_manifests(all_records, manifest_dir)
    for split_name, p in paths.items():
        logger.info("Manifest written: %s -> %s", split_name, p)

    logger.info("Done! Manifests in %s", manifest_dir)


if __name__ == "__main__":
    main()
