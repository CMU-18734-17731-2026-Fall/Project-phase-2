#!/usr/bin/env python3
"""Validate the Phase 2/3 Codabench submission format."""

import argparse
import csv
import io
import math
import sys
import zipfile
from pathlib import Path


EXPECTED_COUNT = 2_000


def parse_prediction_text(text, expected_count=EXPECTED_COUNT):
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("prediction.csv contains a blank row")

    scores = []
    reader = csv.reader(io.StringIO(text))
    for line_number, row in enumerate(reader, start=1):
        if len(row) != 1:
            raise ValueError(
                f"line {line_number} must contain exactly one value, found {len(row)}"
            )
        try:
            value = float(row[0].strip())
        except ValueError as error:
            raise ValueError(
                f"line {line_number} is not a numeric score: {row[0]!r}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"line {line_number} is not finite: {value}")
        scores.append(value)

    if len(scores) != expected_count:
        raise ValueError(
            f"expected {expected_count} scores, found {len(scores)}"
        )
    return scores


def validate_archive(archive_path, phase, expected_count=EXPECTED_COUNT):
    expected_name = f"{phase}/prediction.csv"
    with zipfile.ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if expected_name not in names:
            raise ValueError(
                f"archive must contain {expected_name}; found: {', '.join(names) or '(empty)'}"
            )
        raw = archive.read(expected_name)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("prediction.csv must be UTF-8 text") from error

    scores = parse_prediction_text(text, expected_count)
    extras = [name for name in names if name != expected_name]
    return scores, extras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--phase", choices=("val", "final"), default="val")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        scores, extras = validate_archive(
            args.archive, args.phase, args.expected_count
        )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Submission validation: FAIL\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Submission validation: PASS ({len(scores)} scores)")
    print(f"Score range: [{min(scores):.6g}, {max(scores):.6g}]")
    if extras:
        print("Warning: extra archive entries will be ignored: " + ", ".join(extras))


if __name__ == "__main__":
    main()

