#!/usr/bin/env python3
"""Build a LaTeX summary document from reviewer text files.

The input review files are expected to live in a directory (default: ./reviews)
and to follow the structure produced by the EVN chair template, i.e. each file
contains blocks of proposals separated by a 100-character line of '=' symbols.
Only proposals with at least one non-empty field (grade, referee comments,
technical review, or time recommended) are included in the summary.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List, Optional, Sequence

SEPARATOR = "=" * 100
FIELD_NAMES = [
    "Grade",
    "Referee comments",
    "Technical review",
    "Time recommended",
]


@dataclass
class ReviewEntry:
    reviewer: str
    source_file: Path
    grade: str = ""
    referee_comments: str = ""
    technical_review: str = ""
    time_recommended: str = ""
    role: Optional[int] = None  # 1 for first reviewer, 2 for second reviewer


@dataclass
class ProposalSummary:
    code: str
    title: str
    pi: str = ""
    networks: str = ""
    wavelengths: str = ""
    reviews: List[ReviewEntry] = field(default_factory=list)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert individual review text files into a LaTeX summary document.",
        epilog=dedent(
            """\
            Examples:
              reviews_to_latex.py -r reviews -o summary.tex
              reviews_to_latex.py -r reviews -a reviewer_assignments.txt -t "EVN Reviews" -V "Draft 3"
            """
        ),
    )
    parser.add_argument(
        "-r",
        "--reviews-dir",
        type=Path,
        default=Path("reviews"),
        help="Directory containing renamed review text files (default: ./reviews).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("review_summary.tex"),
        help="Path for the generated LaTeX file (default: ./review_summary.tex).",
    )
    parser.add_argument(
        "-t",
        "--title",
        default="EVN Review Summary",
        help="Title for the LaTeX document.",
    )
    parser.add_argument(
        "-V",
        "--version",
        default="",
        help="Optional version string to display beneath the title.",
    )
    parser.add_argument(
        "-a",
        "--assignments",
        type=Path,
        default=None,
        help="Optional reviewer assignment summary file to tag first/second reviewers.",
    )
    return parser.parse_args(argv)


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in the supplied text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: replacements[match.group()], text)


def reviewer_initials(name: str) -> str:
    """Return uppercase initials derived from the reviewer name."""
    tokens = [token for token in name.replace("_", " ").split() if token]
    initials = "".join(token[0].upper() for token in tokens if token[0].isalpha())
    return initials or "?"


def parse_numeric_grade(value: str) -> Optional[float]:
    """Extract the first numeric token from the grade string."""
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def normalise_person_name(value: str) -> str:
    """Canonical lowercase name for matching assignments."""
    return " ".join(value.lower().split())


def normalise_reviewer_from_filename(path: Path) -> str:
    """Infer the reviewer name from the filename `PREFIX_Name_Surname.ext`."""
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        _, reviewer = parts
        return reviewer.replace("_", " ").strip() or "Unknown reviewer"
    return "Unknown reviewer"


def role_superscript(role: Optional[int]) -> str:
    """Return latex superscript for reviewer role."""
    if role == 1:
        return r"\textsuperscript{*1}"
    if role == 2:
        return r"\textsuperscript{*2}"
    return ""


def load_assignments(path: Path) -> Dict[str, Dict[str, int]]:
    """Parse reviewer assignment summary to map proposal codes to reviewer roles."""
    mapping: Dict[str, Dict[str, int]] = {}
    data = path.read_text(encoding="utf-8")
    if not data.strip():
        return mapping

    lines = data.splitlines()
    if not lines:
        return mapping

    header_line = lines[0].strip().lower()
    if "," in header_line and "proposal" in header_line:
        reader = csv.DictReader(io.StringIO(data))
        for row in reader:
            row_lower = {key.lower(): (value or "").strip() for key, value in row.items()}
            code = row_lower.get("proposal", "")
            if not code:
                continue
            code_map = mapping.setdefault(code, {})
            first = row_lower.get("first reviewer", "")
            second = row_lower.get("second reviewer", "")
            if first:
                code_map[normalise_person_name(first)] = 1
            if second:
                code_map[normalise_person_name(second)] = 2
        return mapping

    pattern = re.compile(r"([A-Z]\d{2}[A-Z]\d{3})\s*\(([^)]+)\)", re.IGNORECASE)
    for line in lines:
        if ":" not in line:
            continue
        name_part, assignments_part = line.split(":", 1)
        reviewer_name = normalise_person_name(name_part.strip())
        if not reviewer_name:
            continue
        for entry in assignments_part.split(","):
            entry = entry.strip()
            if not entry:
                continue
            match = pattern.search(entry)
            if not match:
                continue
            code = match.group(1).strip()
            role_label = match.group(2).strip().lower()
            if "first" in role_label:
                role = 1
            elif "second" in role_label:
                role = 2
            else:
                continue
            code_map = mapping.setdefault(code, {})
            code_map[reviewer_name] = role
    return mapping


def parse_value_block(lines: List[str], start_index: int) -> tuple[str, str, int]:
    """Return (label, value, next_index) for the block starting at start_index."""
    raw_line = lines[start_index]
    label, value = raw_line.split(":", 1)
    label = label.strip()
    value_lines: List[str] = [value.strip()]
    index = start_index + 1
    while index < len(lines):
        candidate = lines[index]
        stripped = candidate.strip()
        if not stripped:
            index += 1
            break
        if any(stripped.startswith(f"{name}:") for name in FIELD_NAMES):
            break
        value_lines.append(stripped)
        index += 1
    cleaned_value = "\n".join(line for line in value_lines if line).strip()
    return label, cleaned_value, index


def parse_review_file(path: Path) -> Dict[str, ProposalSummary]:
    """Parse a single review file into proposal summaries keyed by proposal code."""
    content = path.read_text(encoding="utf-8")
    blocks = [block.strip("\n") for block in content.split(SEPARATOR) if block.strip()]
    summaries: Dict[str, ProposalSummary] = {}

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines()]
        if not lines:
            continue

        header = lines[0]
        if not header.strip():
            continue

        exp = header[0:22].strip()
        pi = header[22:45].strip()
        networks = header[45:72].strip()
        wavelengths = header[72:].strip()
        title = lines[1].strip() if len(lines) > 1 else ""

        field_values = {name: "" for name in FIELD_NAMES}
        index = 2
        while index < len(lines):
            current = lines[index]
            stripped = current.strip()
            if not stripped:
                index += 1
                continue
            if ":" not in stripped:
                index += 1
                continue

            label = stripped.split(":", 1)[0].strip()
            if label not in FIELD_NAMES:
                index += 1
                continue

            label, value, index = parse_value_block(lines, index)
            field_values[label] = value

        if not any(field_values[name] for name in FIELD_NAMES):
            continue  # Skip blocks without substantive content.

        if exp not in summaries:
            summaries[exp] = ProposalSummary(
                code=exp,
                title=title,
                pi=pi,
                networks=networks,
                wavelengths=wavelengths,
                reviews=[],
            )

        reviewer_name = normalise_reviewer_from_filename(path)
        review_entry = ReviewEntry(
            reviewer=reviewer_name,
            source_file=path,
            grade=field_values["Grade"],
            referee_comments=field_values["Referee comments"],
            technical_review=field_values["Technical review"],
            time_recommended=field_values["Time recommended"],
        )
        summaries[exp].reviews.append(review_entry)

    return summaries


def merge_summaries(files: Iterable[Path]) -> Dict[str, ProposalSummary]:
    """Merge per-file summaries into a combined dictionary keyed by proposal code."""
    combined: Dict[str, ProposalSummary] = {}
    for file_path in files:
        file_summaries = parse_review_file(file_path)
        for code, summary in file_summaries.items():
            if code not in combined:
                combined[code] = summary
            else:
                dest = combined[code]
                if not dest.title and summary.title:
                    dest.title = summary.title
                if not dest.pi and summary.pi:
                    dest.pi = summary.pi
                if not dest.networks and summary.networks:
                    dest.networks = summary.networks
                if not dest.wavelengths and summary.wavelengths:
                    dest.wavelengths = summary.wavelengths
                dest.reviews.extend(summary.reviews)
    return combined


def apply_assignments(
    summaries: Dict[str, ProposalSummary], assignments: Dict[str, Dict[str, int]]
) -> None:
    """Annotate review entries with assignment roles where available."""
    for code, summary in summaries.items():
        reviewers_map = assignments.get(code)
        if not reviewers_map:
            continue
        for review in summary.reviews:
            norm = normalise_person_name(review.reviewer)
            role = reviewers_map.get(norm)
            if role is not None:
                review.role = role


def format_review_block(review: ReviewEntry) -> str:
    """Return LaTeX formatted block for a single review entry."""
    parts: List[str] = []
    reviewer_label = latex_escape(review.reviewer)
    supers = role_superscript(review.role)
    parts.append(f"\\subsection*{{Reviewer: {reviewer_label}{supers}}}")

    if review.grade:
        parts.append(f"\\textbf{{Grade:}} {latex_escape(review.grade)}\\\\")
    if review.time_recommended:
        parts.append(
            f"\\textbf{{Time recommended:}} {latex_escape(review.time_recommended)}\\\\"
        )

    if review.referee_comments:
        parts.append("\\textbf{Referee comments}")
        comments = latex_escape(review.referee_comments).replace("\n", "\\\\\n")
        parts.append("\\begin{quote}")
        parts.append(comments)
        parts.append("\\end{quote}")

    if review.technical_review:
        parts.append("\\textbf{Technical review}")
        tech = latex_escape(review.technical_review).replace("\n", "\\\\\n")
        parts.append("\\begin{quote}")
        parts.append(tech)
        parts.append("\\end{quote}")

    source_path = latex_escape(str(review.source_file))
    parts.append(f"\\textit{{Source file:}} {source_path}")
    return "\n".join(parts)


def build_latex_document(
    summaries: Dict[str, ProposalSummary], title: str, version: str = ""
) -> str:
    """Render the combined summaries into a LaTeX document string."""
    preamble = [
        r"\documentclass[twocolumn]{article}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage{enumitem}",
        r"\usepackage{cuted}",
        r"\usepackage{titling}",
        r"\setlength{\droptitle}{-1.5em}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\setlist{nosep}",
        r"\begin{document}",
        rf"\title{{{latex_escape(title)}}}",
    ]

    date_parts: List[str] = []
    if version:
        date_parts.append(f"Version~{latex_escape(version)}")
    date_parts.append(r"\today")
    date_content = r" \\ ".join(date_parts)

    preamble.extend(
        [
            rf"\date{{{date_content}}}",
            r"\maketitle",
        ]
    )
    body: List[str] = []

    for idx, code in enumerate(sorted(summaries.keys())):
        summary = summaries[code]
        grade_entries = [
            (reviewer_initials(review.reviewer), review.grade.strip(), review.role)
            for review in summary.reviews
            if review.grade.strip()
        ]
        if grade_entries:
            sorted_entries = sorted(grade_entries, key=lambda item: item[0])
            header_cells: List[str] = []
            values: List[str] = []
            numeric_values: List[float] = []
            for initials, grade, role in sorted_entries:
                header_cells.append(f"{latex_escape(initials)}{role_superscript(role)}")
                values.append(grade)
                numeric = parse_numeric_grade(grade)
                if numeric is not None:
                    numeric_values.append(numeric)
            average_value = (
                f"{sum(numeric_values) / len(numeric_values):.2f}"
                if numeric_values
                else "N/A"
            )
            header_cells.append(latex_escape("Average"))
            values.append(average_value)

            header_row = " & ".join(header_cells)
            value_row_parts = []
            last_index = len(values) - 1
            for idx_col, value in enumerate(values):
                escaped = latex_escape(value)
                if idx_col == last_index:
                    escaped = f"\\textbf{{{escaped}}}"
                value_row_parts.append(escaped)
            value_row = " & ".join(value_row_parts)

            table_lines = [
                "\\centering",
                f"\\begin{{tabular}}{{{'c' * len(header_cells)}}}",
                header_row + r"\\",
                r"\hline",
                value_row + r"\\",
                "\\end{tabular}",
            ]
        else:
            table_lines = []

        if idx > 0:
            body.append("\\clearpage")

        section_title = f"{summary.code}: {summary.title}" if summary.title else summary.code
        body.append("\\begin{strip}")
        body.append(f"\\section*{{{latex_escape(section_title)}}}")
        if summary.pi:
            body.append(f"\\textbf{{PI:}} {latex_escape(summary.pi)}\\\\")
        if summary.networks:
            body.append(f"\\textbf{{Networks:}} {latex_escape(summary.networks)}\\\\")
        if summary.wavelengths:
            body.append(
                f"\\textbf{{Requested wavelengths:}} {latex_escape(summary.wavelengths)}\\\\"
            )
        if table_lines:
            body.append("")
            body.extend(table_lines)
        if table_lines:
            body.append("")
        body.append("\\end{strip}")
        body.append("")

        if not summary.reviews:
            body.append("\\textit{No reviews available.}")
            continue

        # Sort reviews by reviewer name for deterministic output.
        for review in sorted(summary.reviews, key=lambda r: r.reviewer.lower()):
            body.append(format_review_block(review))
            body.append("")  # Blank line between reviews

    footer = [r"\end{document}"]
    parts: List[str] = []
    parts.extend(preamble)
    parts.extend(body)
    parts.extend(footer)
    return "\n".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    reviews_dir = args.reviews_dir
    if not reviews_dir.exists() or not reviews_dir.is_dir():
        print(f"Reviews directory not found: {reviews_dir}", file=sys.stderr)
        return 1

    review_files = sorted(reviews_dir.glob("*.txt"))
    if not review_files:
        print(f"No .txt review files found in {reviews_dir}", file=sys.stderr)
        return 1

    combined = merge_summaries(review_files)
    if not combined:
        print("No proposal data extracted from review files.", file=sys.stderr)
        return 1

    assignment_map: Dict[str, Dict[str, int]] = {}
    if args.assignments:
        if not args.assignments.is_file():
            print(f"Assignments file not found: {args.assignments}", file=sys.stderr)
            return 1
        assignment_map = load_assignments(args.assignments)
    if assignment_map:
        apply_assignments(combined, assignment_map)

    latex_content = build_latex_document(combined, args.title, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_content, encoding="utf-8")
    print(f"Wrote LaTeX summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
