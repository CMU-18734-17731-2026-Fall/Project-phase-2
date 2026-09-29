#!/usr/bin/env python3
"""Create and validate a Codabench prediction archive."""

import argparse
import sys
import zipfile
from pathlib import Path

from validate_submission import parse_prediction_text, validate_archive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("val", "final"), default="val")
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    prediction_path = args.prediction or Path("data") / args.phase / "prediction.csv"
    output_path = args.output or Path(f"project_phase2_{args.phase}_submission.zip")

    try:
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
        text = prediction_path.read_text(encoding="utf-8")
        parse_prediction_text(text)
    except (OSError, ValueError) as error:
        print(f"Submission creation: FAIL\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    archive_name = f"{args.phase}/prediction.csv"
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(archive_name, text)

    scores, _ = validate_archive(output_path, args.phase)
    print(f"Created {output_path} with {len(scores)} scores")


if __name__ == "__main__":
    main()
