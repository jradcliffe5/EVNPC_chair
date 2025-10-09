#!/usr/bin/env python3
"""Rename chair review files based on CSV records.

The script expects the CSV to provide `Name` and `Surname` columns (as in the
Google Forms export). For each row it looks in the source directory for a file
matching the pattern:

    <input_name> - Name Surname.txt

and moves it to the destination directory with the new filename:

    <prefix>_Name_Surname.txt

Both the original input name and the target prefix are case-sensitive strings,
while reviewer names are matched case-insensitively (with redundant whitespace
collapsed). The original file extension is preserved.

Example:
    python rename_reviews_from_csv.py pc_chair/EVN_PC_review_submission.csv \
        --prefix E25A001 --source-dir "Copy of EVN.../Review submission (File responses)" \
        --dest-dir pc_chair/reviews
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename existing review files from '<input> - Name Surname.ext' to "
            "'<prefix>_Name_Surname.ext'."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV export (e.g. EVN_PC_review_submission.csv).",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Text to prepend to each output filename (replaces <input> placeholder).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the downloaded review files. "
            "Defaults to the CSV's parent directory."
        ),
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where renamed files will be placed (created if missing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned moves without changing any files.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Silently continue when a referenced file cannot be located.",
    )
    return parser.parse_args(argv)


def sanitized_component(value: str) -> str:
    """Return a filesystem-safe name component derived from the input."""
    stripped = value.strip()
    if not stripped:
        return "Unknown"
    filtered = "".join(ch for ch in stripped if ch.isalnum())
    return filtered or "Unknown"


def normalize_name(value: str) -> str:
    """Normalise a name for comparison (case-insensitive, single spaces)."""
    return " ".join(value.lower().split())


def find_submission_file(
    name: str,
    surname: str,
    source_dir: Path,
) -> Optional[Path]:
    """Locate a file matching '* - Name Surname.*' in the source directory."""
    target_tail = normalize_name(f"- {name} {surname}")
    # Search in deterministic order to make behaviour predictable.
    for candidate in sorted(source_dir.rglob("*")):
        if not candidate.is_file():
            continue
        stem = candidate.stem
        # Preserve the part before the extension; multi-dot names are fine.
        stem_normalized = normalize_name(stem.replace("_", " "))
        if stem_normalized.endswith(target_tail):
            return candidate
    return None


def unique_destination(path: Path) -> Path:
    """Return a unique path by appending a numeric suffix if required."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def build_new_name(prefix: str, name: str, surname: str, suffix: str) -> str:
    """Construct the new filename from the prefix and reviewer name."""
    components = [prefix]
    for value in (name, surname):
        components.append(sanitized_component(value))
    return "_".join(components) + suffix


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    csv_path: Path = args.csv_path
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    source_dir: Path = args.source_dir or csv_path.parent
    dest_dir: Path = args.dest_dir
    dry_run: bool = args.dry_run

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1
    dest_dir.mkdir(parents=True, exist_ok=True)

    missing: list[Tuple[str, str, str]] = []
    processed = 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Name", "").strip()
            surname = row.get("Surname", "").strip()

            if not name and not surname:
                if not args.skip_missing:
                    missing.append(("", "", "Missing Name and Surname in CSV row."))
                continue

            located = find_submission_file(name, surname, source_dir)
            if located is None:
                if not args.skip_missing:
                    missing.append((name, surname, "Matching file not found."))
                continue

            suffix = located.suffix or ".txt"
            new_name = build_new_name(args.prefix, name, surname, suffix)
            destination = unique_destination(dest_dir / new_name)

            if dry_run:
                print(f"[DRY-RUN] {located} -> {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                located.replace(destination)
                print(f"Moved: {located} -> {destination}")
            processed += 1

    if missing:
        print("\nUnable to process:", file=sys.stderr)
        for name, surname, reason in missing:
            label = (f"{name} {surname}".strip() or "Unknown reviewer").strip()
            if reason:
                print(f"  {label}: {reason}", file=sys.stderr)
            else:
                print(f"  {label}", file=sys.stderr)
        return 2 if not dry_run else 0

    if processed == 0:
        print("No submissions processed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
