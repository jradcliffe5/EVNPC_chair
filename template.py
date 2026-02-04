#!/usr/bin/env python3
"""Render EVNPC chair review templates from ampersand-delimited input.

This is a Python port of the original `template.prl` Perl script.
"""

import sys
from typing import Iterable, List, Optional

SEPARATOR = "=" * 100
LINE_TEMPLATE = "{exp:<22} {pi:<22}  {nets:<25}  {lambda_:<38}"
TITLE_WIDTH = 99
EXPECTED_FIELDS = 8


def normalize_fields(parts: List[str]) -> List[str]:
    """Pad or trim the split fields to match the expected field count."""
    if len(parts) >= EXPECTED_FIELDS:
        return parts[:EXPECTED_FIELDS]
    return parts + [""] * (EXPECTED_FIELDS - len(parts))


def render_record(
    exp: str,
    pi: str,
    nets: str,
    lambda_: str,
    title: str,
    first_reviewer: Optional[str] = None,
    second_reviewer: Optional[str] = None,
    additional_reviewers: Optional[List[str]] = None,
) -> Iterable[str]:
    """Yield formatted lines for a single record."""
    yield SEPARATOR
    yield LINE_TEMPLATE.format(exp=exp, pi=pi, nets=nets, lambda_=lambda_)
    yield f"{title:<{TITLE_WIDTH}}"
    yield ""
    yield "Grade:"
    yield ""
    yield "Referee comments:"
    yield "Strengths:"
    yield "Weaknesses:"
    yield ""
    yield "Technical review:"
    yield ""
    yield "Time recommended:"


def main() -> None:
    for raw_line in sys.stdin:
        stripped = raw_line.rstrip("\n")
        if not stripped:
            continue
        fields = normalize_fields(stripped.split("&"))
        (
            _,
            _,
            exp,
            pi,
            title,
            _,
            nets,
            lambda_,
        ) = fields

        for line in render_record(exp, pi, nets, lambda_, title):
            print(line)


if __name__ == "__main__":
    main()
