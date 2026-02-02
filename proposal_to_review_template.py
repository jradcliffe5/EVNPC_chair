#!/usr/bin/env python3
"""Parse EVN proposal PDFs and render chair summaries.

Usage
=====
python proposal_to_review_template.py [-h] [-p PDF_DIR] [-o OUTPUT] [-m PC_MEMBERS] [-a ASSIGNMENTS] [pdfs ...]

positional arguments:
  pdfs                  Specific PDF files to process. Defaults to all PDFs in --pdf-dir.

optional arguments:
  -h, --help            Show this help message and exit.
  -p PDF_DIR, --pdf-dir PDF_DIR
                        Directory containing proposal PDFs (required if no PDF files are listed).
  -o OUTPUT, --output OUTPUT
                        Write the formatted output to this file instead of stdout.
  -m PC_MEMBERS, --pc-members PC_MEMBERS
                        File containing EVN PC members (one per line) to auto-assign reviewers. If a member
                        should always referee a specific proposal, append tokens like CODE#1 (first reviewer)
                        or CODE#2 (second reviewer) after their name, e.g. "Jane Smith E25A001#1 E25A010#2".
  -a ASSIGNMENTS, --assignments ASSIGNMENTS
                        File to store reviewer assignments (defaults to reviewer_assignments.txt when --pc-members is used).
  --reviewers-per-proposal COUNT
                        Total reviewers to assign per proposal (minimum 2, default 2).
  --max-per-member MAX  Maximum number of proposals assigned to any single PC member.
  --max-first-per-member COUNT
                        Maximum number of first-reviewer slots per PC member.
  --max-second-per-member COUNT
                        Maximum number of second-reviewer slots per PC member.
  --member-summary FILE Write a per-member HTML assignment table to FILE.
  --conflicts-file FILE Load additional conflicts from FILE (same format as reviewer_assignments appendix).

Examples:
  python proposal_to_review_template.py -p test_proposals -m EVN_pc_members.txt
  python proposal_to_review_template.py proposal.pdf another.pdf -m EVN_pc_members.txt -o summaries.txt
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import defaultdict

from template import render_record

# Match experiment identifiers and waveband tokens in the PDF text.
PROPOSAL_CODE_RE = re.compile(r"\b[EG]\d{2}[A-Z]\d{3}\b")
WAVEBAND_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm|m|GHz|MHz)", re.IGNORECASE)


def generate_role_labels(count: int) -> List[str]:
    """Return role labels for the ordered reviewer slots."""
    labels: List[str] = []
    for idx in range(count):
        if idx == 0:
            labels.append("First Reviewer")
        elif idx == 1:
            labels.append("Second Reviewer")
        else:
            labels.append(f"Additional Reviewer {idx - 1}")
    return labels

# Known network keywords to capture from the summary table.
NETWORK_KEYWORDS = {
    "EVN",
    "MERLIN",
    "VLBA",
    "MeerKAT",
    "uGMRT",
    "NRAO",
    "LBA",
    "EAVN",
    "KVN",
    "GMVA",
    "ATCA",
    "Other",
    "JVN",
    "JNET",
    "e-MERLIN",
}
SUMMARY_STOP_PREFIXES = (
    "no phd",
    "students involved",
    "student",
    "is this",
    "linked proposal",
    "relevant previous",
    "observation dependencies",
    "aggregate correlator",
    "processor information",
    "print view prepared",
    "scientific category",
    "scheduling assistance",
    "rapid response science",
)
SUMMARY_SKIP_PHRASES = (
    "Observation Number of Network",
    "number      targets",
    "Aggregate Correlator e-EVN",
    "Out-of-",
)

# Lowercase salutations and honorifics to strip from normalised names.
TITLE_PREFIXES = {
    "dr",
    "prof",
    "professor",
    "mr",
    "mrs",
    "ms",
    "miss",
    "sir",
    "madam",
}


def normalise_name(name: str) -> str:
    """Return a lowercase, punctuation-free version of a personal name."""
    cleaned = re.sub(r"\s+", " ", name).strip().lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    tokens = cleaned.split()
    filtered_tokens = [token for token in tokens if token not in TITLE_PREFIXES]
    return " ".join(filtered_tokens) if filtered_tokens else cleaned


class PdfExtractError(RuntimeError):
    """Raised when pdftotext is unavailable or fails."""


def extract_pdf_lines(path: Path) -> List[str]:
    """Return the PDF content as layout-preserving text lines via pdftotext."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:  # pragma: no cover - poppler not installed
        raise PdfExtractError("pdftotext command not found; please install Poppler.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise PdfExtractError(f"pdftotext failed for {path.name}: {stderr.strip()}") from exc
    text = result.stdout.decode("utf-8", errors="ignore")
    return text.splitlines()


def find_experiment_and_title(lines: Sequence[str], fallback: str) -> tuple[str, str, Optional[str]]:
    """Locate the proposal code, title, and inline PI hint from text lines."""
    exp = None
    exp_idx = 0
    pi_hint = None
    for idx, line in enumerate(lines):
        match = PROPOSAL_CODE_RE.search(line)
        if match:
            exp = match.group()
            exp_idx = idx
            pi_hint = line[: match.start()].strip() or None
            break
    if exp is None:
        return fallback, fallback, None

    title_parts: List[str] = []
    for line in lines[exp_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if title_parts:
                break
            continue
        if stripped.lower().startswith("abstract"):
            break
        title_parts.append(stripped)
    title = " ".join(title_parts).strip() or fallback
    return exp, title, pi_hint


def parse_applicants(lines: Sequence[str]) -> List[dict[str, str]]:
    """Parse the Applicants table into a list of dictionaries."""
    try:
        start = lines.index("Applicants")
    except ValueError:
        return []

    rows: List[dict[str, str]] = []
    current: Optional[dict[str, str]] = None
    keys = ["name", "affiliation", "email", "country", "potential"]
    skip_tokens = {"observer", "potential"}

    i = start + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "Contact Author":
            if current:
                rows.append(current)
            break
        if stripped.lower() in skip_tokens:
            i += 1
            continue
        if "Name" in stripped and "Affiliation" in stripped and "Potential" in stripped:
            i += 1
            continue

        parts = [p for p in re.split(r"\s{2,}", line.rstrip()) if p]
        if not parts:
            i += 1
            continue

        if len(parts) >= len(keys):
            if current:
                rows.append(current)
            current = {keys[idx]: parts[idx].strip() for idx in range(len(keys))}
        else:
            if current is None:
                current = {key: "" for key in keys}
            if len(parts) == 1:
                current["affiliation"] = " ".join(filter(None, [current.get("affiliation"), parts[0].strip()]))
            elif len(parts) == 2:
                current["affiliation"] = " ".join(filter(None, [current.get("affiliation"), parts[0].strip()]))
                current["email"] = " ".join(filter(None, [current.get("email"), parts[1].strip()]))
            else:
                current["potential"] = " ".join(filter(None, [current.get("potential"), " ".join(parts).strip()]))
        i += 1

    if current and (not rows or current is not rows[-1]):
        rows.append(current)
    return rows


def parse_contact_name(lines: Sequence[str]) -> Optional[str]:
    """Extract the contact author name from the dedicated section."""
    try:
        start = lines.index("Contact Author")
    except ValueError:
        return None

    for idx in range(start + 1, min(start + 10, len(lines))):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        parts = [p for p in re.split(r"\s{2,}", stripped) if p]
        if not parts:
            continue
        if parts[0].lower() == "name" and len(parts) > 1:
            return parts[1].strip()
    return None


def normalise_waveband(value: str, unit: str) -> str:
    """Standardise waveband labels with consistent units."""
    unit_map = {"cm": "cm", "mm": "mm", "m": "m", "ghz": "GHz", "mhz": "MHz"}
    normalised_unit = unit_map.get(unit.lower(), unit.upper())
    clean_value = value[:-2] if value.endswith(".0") else value
    return f"{clean_value} {normalised_unit}"


def extract_wavebands(segment: str, collected: List[str]) -> None:
    """Collect unique waveband entries from a summary segment."""
    for value, unit in WAVEBAND_RE.findall(segment):
        label = normalise_waveband(value, unit)
        if label not in collected:
            collected.append(label)


def segments_to_network_tokens(segments: Sequence[str]) -> List[str]:
    """Flatten multi-line network segments into distinct network tokens."""
    tokens: List[str] = []
    buffer = ""
    for segment in segments:
        segment = segment.strip()
        if not segment or segment in {",", ";"}:
            continue
        parts = [p.strip() for p in segment.split(",") if p.strip()]
        trailing_comma = segment.endswith(",")
        if not parts:
            if trailing_comma and buffer:
                if keep_network(buffer) and buffer not in tokens:
                    tokens.append(buffer)
                buffer = ""
            continue
        for idx, part in enumerate(parts):
            if buffer:
                if part.upper() == "NRAO" and buffer.lower().endswith("other"):
                    buffer = f"{buffer} {part}"
                elif part.islower() or part == part.lower():
                    buffer = f"{buffer} {part}".strip()
                else:
                    if keep_network(buffer) and buffer not in tokens:
                        tokens.append(buffer)
                    buffer = part
            else:
                buffer = part
            if idx < len(parts) - 1 or trailing_comma:
                if keep_network(buffer) and buffer not in tokens:
                    tokens.append(buffer)
                buffer = ""
    if buffer and keep_network(buffer) and buffer not in tokens:
        tokens.append(buffer)
    return tokens


def keep_network(token: str) -> bool:
    """Return True if the token contains a recognized network keyword."""
    upper = token.upper()
    return any(keyword in upper for keyword in NETWORK_KEYWORDS)


def strip_numbers(segment: str) -> str:
    """Remove numeric values and waveband text to leave network fragments."""
    cleaned = segment
    for value, unit in WAVEBAND_RE.findall(segment):
        cleaned = cleaned.replace(f"{value}{unit}", "")
        cleaned = cleaned.replace(f"{value} {unit}", "")
    cleaned = re.sub(r"\d+(?:\.\d+)?", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_summary(lines: Sequence[str]) -> tuple[List[str], List[str]]:
    """Pull network list and requested wavebands from the summary table."""
    try:
        start = lines.index("Summary of Observations")
    except ValueError:
        return [], []

    networks: List[str] = []
    wavebands: List[str] = []
    current_segments: List[str] = []

    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(lower.startswith(prefix) for prefix in SUMMARY_STOP_PREFIXES):
            break
        if any(phrase in stripped for phrase in SUMMARY_SKIP_PHRASES):
            continue

        parts = [p for p in re.split(r"\s{2,}", stripped) if p]
        new_row = parts and parts[0].isdigit()
        data_parts: Iterable[str] = parts[2:] if new_row else parts

        if new_row and current_segments:
            for token in segments_to_network_tokens(current_segments):
                if token not in networks:
                    networks.append(token)
            current_segments = []

        for idx, segment in enumerate(data_parts):
            if not segment.strip():
                continue
            extract_wavebands(segment, wavebands)
            if new_row and idx > 0:
                continue
            cleaned = strip_numbers(segment)
            if cleaned:
                current_segments.append(cleaned)

    if current_segments:
        for token in segments_to_network_tokens(current_segments):
            if token not in networks:
                networks.append(token)

    return networks, wavebands


def load_pc_members(path: Path) -> Tuple[List[str], Dict[str, Dict[str, List[str]]], Dict[str, str], Set[str]]:
    """Read PC member entries and return names, fixed reviewer preferences, email mapping, and chair markers.

    Appending `*` to any part of a member's name marks them as a chair who should receive leftover assignments.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read PC members file: {path}") from exc

    members: List[str] = []
    fixed: Dict[str, Dict[str, List[str]]] = {}
    emails: Dict[str, str] = {}
    chairs: Set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        name_tokens: List[str] = []
        first_fixed: List[str] = []
        second_fixed: List[str] = []
        email: Optional[str] = None
        is_chair = False
        for token in tokens:
            if "#" in token:
                try:
                    proposal_code, slot = token.split("#", 1)
                except ValueError:
                    continue
                proposal_code = proposal_code.strip()
                if not proposal_code:
                    continue
                if slot == "1":
                    first_fixed.append(proposal_code)
                elif slot == "2":
                    second_fixed.append(proposal_code)
            elif email is None and "@" in token:
                cleaned_email = token.strip("<>[](){};,")
                if cleaned_email:
                    email = cleaned_email
            else:
                chair_token = "*" in token
                cleaned_token = token.replace("*", "")
                if cleaned_token:
                    name_tokens.append(cleaned_token)
                if chair_token:
                    is_chair = True
        name = " ".join(name_tokens).strip()
        if not name:
            continue
        members.append(name)
        if is_chair:
            chairs.add(name)
        if first_fixed or second_fixed:
            fixed[name] = {
                "first": first_fixed,
                "second": second_fixed,
            }
        if email:
            emails[name] = email

    if not members:
        raise ValueError(f"No PC members found in {path}")
    return members, fixed, emails, chairs


def load_conflicts_file(path: Path) -> Dict[str, Set[str]]:
    """Parse a conflicts file formatted like the reviewer_assignments appendix."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read conflicts file: {path}") from exc

    conflicts: Dict[str, Set[str]] = defaultdict(set)
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "conflicts:":
            continue
        if ":" not in line:
            continue
        proposal_code, names = line.split(":", 1)
        proposal_code = proposal_code.strip()
        if not proposal_code:
            continue
        entries = [entry.strip() for entry in names.split(",") if entry.strip()]
        if not entries or (len(entries) == 1 and entries[0].lower() == "none"):
            continue
        for entry in entries:
            conflicts[proposal_code].add(entry)
    return conflicts


def assign_reviewers(
    proposals: List[Dict[str, Any]],
    members: Sequence[str],
    reviewers_per_proposal: int,
    max_per_member: Optional[int] = None,
    fixed_preferences: Optional[Dict[str, Dict[str, List[str]]]] = None,
    max_first_per_member: Optional[int] = None,
    max_second_per_member: Optional[int] = None,
    chair_members: Optional[Set[str]] = None,
    manual_conflicts: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """Assign reviewers while balancing load, respecting per-role limits, and prioritising chairs for leftovers."""
    if not members:
        raise ValueError("Cannot assign reviewers without PC members.")
    if reviewers_per_proposal < 2:
        raise ValueError("Each proposal must have at least two reviewers.")
    if max_per_member is not None and max_per_member <= 0:
        raise ValueError("Maximum proposals per member must be positive.")
    if max_first_per_member is not None and max_first_per_member <= 0:
        raise ValueError("Maximum first-reviewer assignments per member must be positive.")
    if max_second_per_member is not None and max_second_per_member <= 0:
        raise ValueError("Maximum second-reviewer assignments per member must be positive.")
    if reviewers_per_proposal > len(members):
        raise ValueError("Not enough PC members to satisfy reviewers-per-proposal.")
    chair_members = chair_members or set()
    member_infos = [
        {
            "name": name,
            "normalised": normalise_name(name),
            "count": 0,
            "first_count": 0,
            "second_count": 0,
            "is_chair": name in chair_members,
            "order": idx,
        }
        for idx, name in enumerate(members)
        if normalise_name(name)
    ]
    if not member_infos:
        raise ValueError("PC member list does not contain valid names.")
    per_member: Dict[str, List[Tuple[str, str]]] = {info["name"]: [] for info in member_infos}
    members_by_name: Dict[str, dict] = {info["name"]: info for info in member_infos}
    members_by_normalised: Dict[str, str] = {
        info["normalised"]: info["name"] for info in member_infos if info["normalised"]
    }
    role_labels = generate_role_labels(reviewers_per_proposal)

    fixed_first_map: Dict[str, str] = {}
    fixed_second_map: Dict[str, str] = {}
    if fixed_preferences:
        for member_name, slots in fixed_preferences.items():
            if member_name not in members_by_name:
                raise ValueError(f"PC member '{member_name}' with fixed assignments is not in the member list.")
            for code in slots.get("first", []):
                if not code:
                    continue
                existing = fixed_first_map.get(code)
                if existing and existing != member_name:
                    raise ValueError(f"Conflicting first reviewer assignment for proposal {code}.")
                fixed_first_map[code] = member_name
            for code in slots.get("second", []):
                if not code:
                    continue
                existing = fixed_second_map.get(code)
                if existing and existing != member_name:
                    raise ValueError(f"Conflicting second reviewer assignment for proposal {code}.")
                fixed_second_map[code] = member_name

    def has_capacity(member: dict, role: str) -> bool:
        if role == "First Reviewer" and max_first_per_member is not None:
            if member["first_count"] >= max_first_per_member:
                return False
        if role == "Second Reviewer" and max_second_per_member is not None:
            if member["second_count"] >= max_second_per_member:
                return False
        return True

    def record_assignment(member: dict, role: str) -> None:
        member["count"] += 1
        if role == "First Reviewer":
            member["first_count"] += 1
        elif role == "Second Reviewer":
            member["second_count"] += 1

    def priority_key(member: dict, role: str) -> Tuple[int, int, int, int]:
        """Return a tuple used to balance role-specific assignments (chairs soak up leftovers)."""
        chair_bias = 0 if member.get("is_chair") else 1
        if role == "First Reviewer":
            return (member["first_count"], chair_bias, member["count"], member["order"])
        if role == "Second Reviewer":
            return (member["second_count"], chair_bias, member["count"], member["order"])
        return (member["count"], chair_bias, member["order"], 0)

    def select_member(excluded: Set[str], already_chosen: Set[str], role: str) -> dict:
        eligible = [
            member
            for member in member_infos
            if member["normalised"] not in excluded
            and member["normalised"] not in already_chosen
            and (max_per_member is None or member["count"] < max_per_member)
            and has_capacity(member, role)
        ]
        if not eligible:
            eligible = [
                member
                for member in member_infos
                if member["normalised"] not in excluded
                and (max_per_member is None or member["count"] < max_per_member)
                and has_capacity(member, role)
            ]
        if not eligible:
            raise ValueError("No available reviewers remaining for assignment within limits.")
        chosen = min(eligible, key=lambda m: priority_key(m, role))
        return chosen

    def apply_fixed_member(
        member_name: str,
        role_idx: int,
        proposal_code: str,
        excluded: Set[str],
        already_chosen: Set[str],
        reviewers: List[Tuple[str, str]],
    ) -> None:
        member = members_by_name.get(member_name)
        if member is None:
            raise ValueError(f"Fixed reviewer '{member_name}' is not a recognised PC member.")
        if member["normalised"] in excluded:
            raise ValueError(
                f"Fixed reviewer '{member_name}' is listed on proposal {proposal_code}, cannot assign."
            )
        if member["normalised"] in already_chosen:
            raise ValueError(
                f"Fixed reviewer '{member_name}' requested multiple slots for proposal {proposal_code}."
            )
        if max_per_member is not None and member["count"] >= max_per_member:
            raise ValueError(
                f"Fixed reviewer '{member_name}' exceeds the maximum assignments ({max_per_member})."
            )
        role = role_labels[role_idx]
        if not has_capacity(member, role):
            limit_label = "first" if role == "First Reviewer" else "second"
            raise ValueError(
                f"Fixed reviewer '{member_name}' exceeds the maximum {limit_label}-reviewer assignments."
            )
        record_assignment(member, role)
        already_chosen.add(member["normalised"])
        reviewers.append((role, member_name))
        per_member[member_name].append((proposal_code, role))

    for proposal in proposals:
        proposal_code = proposal["exp"]
        participants: Set[str] = set(proposal.get("participants", set()))
        excluded = set(participants)
        # Capture everyone we exclude so the CSV can describe conflicts explicitly.
        conflicts: Set[str] = set()
        if participants:
            for member in member_infos:
                norm = member["normalised"]
                if norm and norm in participants:
                    conflicts.add(member["name"])
        text_blob = proposal.get("normalised_text", "")
        if text_blob:
            for member in member_infos:
                norm = member["normalised"]
                if norm and f" {norm} " in text_blob:
                    excluded.add(norm)
                    conflicts.add(member["name"])
        if manual_conflicts:
            extra_conflicts = manual_conflicts.get(proposal_code, set())
            for entry in extra_conflicts:
                resolved_name = entry.strip()
                norm = normalise_name(resolved_name)
                if norm:
                    excluded.add(norm)
                    resolved_name = members_by_normalised.get(norm, resolved_name)
                if resolved_name:
                    conflicts.add(resolved_name)
        chosen: Set[str] = set()
        reviewers: List[Tuple[str, str]] = []
        fixed_slots: List[Optional[str]] = [None] * reviewers_per_proposal
        fixed_first = fixed_first_map.get(proposal_code)
        fixed_second = fixed_second_map.get(proposal_code)
        if fixed_first:
            fixed_slots[0] = fixed_first
        if fixed_second:
            if reviewers_per_proposal < 2:
                raise ValueError(
                    f"Cannot assign second reviewer for {proposal_code} when reviewers-per-proposal < 2."
                )
            if fixed_first and fixed_second == fixed_first:
                raise ValueError(f"Member '{fixed_first}' cannot be both first and second reviewer for {proposal_code}.")
            fixed_slots[1] = fixed_second

        for idx in range(reviewers_per_proposal):
            fixed_member = fixed_slots[idx]
            role = role_labels[idx]
            if fixed_member:
                apply_fixed_member(fixed_member, idx, proposal_code, excluded, chosen, reviewers)
                continue
            member = select_member(excluded, chosen, role)
            chosen.add(member["normalised"])
            record_assignment(member, role)
            reviewers.append((role, member["name"]))
            per_member[member["name"]].append((proposal_code, role))

        proposal["reviewers"] = reviewers
        proposal["first_reviewer"] = reviewers[0][1]
        proposal["second_reviewer"] = reviewers[1][1]
        proposal["conflicts"] = sorted(conflicts)

    return per_member


def write_assignments(proposals: Sequence[Dict[str, Any]], destination: Path, roles: Sequence[str]) -> None:
    """Persist reviewer assignments to a CSV file and append a conflicts appendix."""
    roles = list(roles)
    max_count = max((len(proposal.get("reviewers", [])) for proposal in proposals), default=0)
    if max_count > len(roles):
        extra_roles = generate_role_labels(max_count)[len(roles):]
        roles.extend(extra_roles)

    header = ["Proposal", *roles]
    csv_lines = [",".join(header)]

    for proposal in proposals:
        row = [proposal["exp"]]
        reviewers = proposal.get("reviewers", [])
        role_map = {role: name for role, name in reviewers}
        for role in roles:
            row.append(role_map.get(role, ""))
        csv_lines.append(",".join(row))

    conflict_lines: List[str] = []
    for proposal in proposals:
        conflicts = proposal.get("conflicts") or []
        if conflicts:
            conflict_lines.append(f"{proposal['exp']}: {', '.join(conflicts)}")
        else:
            conflict_lines.append(f"{proposal['exp']}: None")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = ["\n".join(csv_lines)]
    if conflict_lines:
        sections.append("Conflicts:")
        sections.append("\n".join(conflict_lines))
    content = "\n\n".join(sections)
    if not content.endswith("\n"):
        content += "\n"
    destination.write_text(content, encoding="utf-8")


def build_reviewer_email_table(assignments: Dict[str, List[Tuple[str, str]]], roles: Sequence[str]) -> str:
    """Return an HTML table summarizing per-reviewer assignments by role."""
    if not assignments:
        return ""

    base_roles: List[str] = []
    extra_roles: List[str] = []
    for role in roles:
        if role.lower().startswith("additional reviewer"):
            extra_roles.append(role)
        else:
            base_roles.append(role)

    headers = ["Reviewer", *base_roles]
    if extra_roles:
        headers.append("Additional Reviewers")

    lines = [
        '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;">',
        "  <thead>",
        "    <tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr>",
        "  </thead>",
        "  <tbody>",
    ]

    for reviewer in sorted(assignments.keys(), key=str.lower):
        row_cells: List[str] = [html.escape(reviewer)]
        slot_map: Dict[str, List[str]] = defaultdict(list)
        for proposal_code, role in assignments[reviewer]:
            slot_map[role].append(proposal_code)
        for role in base_roles:
            entries = slot_map.get(role)
            row_cells.append(format_assignment_entries(entries))
        if extra_roles:
            combined: List[str] = []
            for role in extra_roles:
                combined.extend(slot_map.get(role, []))
            row_cells.append(format_assignment_entries(combined))
        lines.append("    <tr>" + "".join(f"<td>{cell}</td>" for cell in row_cells) + "</tr>")

    lines.append("  </tbody>")
    lines.append("</table>")

    return "\n".join(lines)


def format_assignment_entries(entries: Optional[Sequence[str]]) -> str:
    """Format assignment codes for a table cell."""
    if not entries:
        return ""
    return ", ".join(html.escape(entry) for entry in entries)


def write_member_summary(assignments: Dict[str, List[Tuple[str, str]]], destination: Path, roles: Sequence[str]) -> None:
    """Persist per-member assignments as an HTML table for easy emailing."""
    table_html = build_reviewer_email_table(assignments, roles)
    if not table_html:
        table_html = "<p>No reviewer assignments available.</p>"

    destination.parent.mkdir(parents=True, exist_ok=True)
    html_content = "\n".join(
        [
            "<!-- Reviewer assignments for Outlook. Paste the table below into the email body. -->",
            table_html,
        ]
    )
    if not html_content.endswith("\n"):
        html_content += "\n"
    destination.write_text(html_content, encoding="utf-8")


def parse_proposal(path: Path) -> Dict[str, Any]:
    """Derive structured proposal metadata from a PDF."""
    lines = extract_pdf_lines(path)
    exp, title, pi_hint = find_experiment_and_title(lines, path.stem)
    normalised_text = f" {normalise_name(' '.join(lines))} "

    applicants = parse_applicants(lines)
    pi = next(
        (row["name"] for row in applicants if "pi" in row.get("potential", "").lower()),
        None,
    )
    if not pi and applicants:
        pi = applicants[0].get("name")
    if not pi:
        pi = pi_hint or parse_contact_name(lines) or "Unknown"

    participants: Set[str] = set()
    for row in applicants:
        name = row.get("name", "")
        normalised = normalise_name(name)
        if normalised:
            participants.add(normalised)

    contact_name = parse_contact_name(lines)
    if contact_name:
        normalised_contact = normalise_name(contact_name)
        if normalised_contact:
            participants.add(normalised_contact)
    normalised_pi = normalise_name(pi)
    if normalised_pi:
        participants.add(normalised_pi)

    networks, wavebands = parse_summary(lines)
    nets = ", ".join(networks)
    lambdas = ", ".join(wavebands)

    return {
        "exp": exp,
        "pi": pi,
        "title": title,
        "nets": nets,
        "lambda": lambdas,
        "participants": participants,
        "normalised_text": normalised_text,
    }


def iter_pdf_paths(supplied: Sequence[Path], pdf_dir: Optional[Path]) -> Iterable[Path]:
    """Yield the PDFs to process, combining explicit files with directory listings."""
    if pdf_dir is not None and not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    if supplied:
        for supplied_path in supplied:
            if supplied_path.is_file():
                yield supplied_path
                continue
            if pdf_dir is not None:
                candidate = supplied_path if supplied_path.is_absolute() else pdf_dir / supplied_path
                if candidate.is_file():
                    yield candidate
                    continue
            raise FileNotFoundError(f"PDF not found: {supplied_path}")
    else:
        if pdf_dir is None:
            raise FileNotFoundError("No PDF directory provided and no PDF files supplied.")
        for path in sorted(pdf_dir.glob("*.pdf")):
            if path.is_file():
                yield path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for rendering templates and assigning reviewers."""
    parser = argparse.ArgumentParser(
        description="Render EVN proposal PDFs in chair template format.",
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="Specific PDF files to process. Defaults to all PDFs in --pdf-dir.",
    )
    parser.add_argument(
        "-p",
        "--pdf-dir",
        type=Path,
        help="Directory containing proposal PDFs (required if no PDF files are listed).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the formatted output to this file instead of stdout.",
    )
    parser.add_argument(
        "-m",
        "--pc-members",
        type=Path,
        help="File containing EVN PC members (`Name Email` per line, optional CODE#slot) to auto-assign reviewers.",
    )
    parser.add_argument(
        "-a",
        "--assignments",
        type=Path,
        help="File to store reviewer assignments (defaults to reviewer_assignments.txt when --pc-members is used).",
    )
    parser.add_argument(
        "--reviewers-per-proposal",
        type=int,
        default=2,
        metavar="COUNT",
        help="Total reviewers to assign per proposal (minimum 2, default 2).",
    )
    parser.add_argument(
        "--max-per-member",
        type=int,
        metavar="MAX",
        help="Maximum number of proposals assigned to any single PC member.",
    )
    parser.add_argument(
        "--max-first-per-member",
        type=int,
        metavar="COUNT",
        help="Maximum number of first reviewer assignments per PC member.",
    )
    parser.add_argument(
        "--max-second-per-member",
        type=int,
        metavar="COUNT",
        help="Maximum number of second reviewer assignments per PC member.",
    )
    parser.add_argument(
        "--member-summary",
        type=Path,
        help="Write a per-member HTML assignment table to this file.",
    )
    parser.add_argument(
        "--conflicts-file",
        type=Path,
        help="Optional file listing per-proposal conflicts (same format as the reviewer assignment appendix).",
    )
    args = parser.parse_args(argv)

    try:
        pdf_paths = list(iter_pdf_paths(args.pdfs, args.pdf_dir))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not pdf_paths:
        print("No proposal PDFs found.", file=sys.stderr)
        return 1

    if (
        args.assignments
        or args.max_per_member
        or args.member_summary
        or args.max_first_per_member
        or args.max_second_per_member
        or args.conflicts_file
    ) and not args.pc_members:
        print("Reviewer-related options require --pc-members to be specified.", file=sys.stderr)
        return 1
    if args.reviewers_per_proposal < 2:
        print("--reviewers-per-proposal must be at least 2.", file=sys.stderr)
        return 1

    proposals: List[Dict[str, Any]] = []

    for path in pdf_paths:
        try:
            proposal = parse_proposal(path)
        except PdfExtractError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 2
        proposal["pdf_path"] = str(path)
        proposals.append(proposal)

    member_assignments: Optional[Dict[str, List[Tuple[str, str]]]] = None
    role_labels: Optional[List[str]] = None
    manual_conflicts: Dict[str, Set[str]] = {}

    if args.conflicts_file:
        try:
            manual_conflicts = load_conflicts_file(args.conflicts_file)
        except (FileNotFoundError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1

    if args.pc_members:
        try:
            members, fixed_preferences, _member_emails, chair_members = load_pc_members(args.pc_members)
        except (FileNotFoundError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        proposal_count = len(proposals)
        member_count = len(members)
        if args.max_first_per_member is not None and member_count * args.max_first_per_member < proposal_count:
            print(
                f"Insufficient first-reviewer capacity: need {proposal_count} slots for {proposal_count} proposals, "
                f"but --max-first-per-member={args.max_first_per_member} with {member_count} members allows only "
                f"{member_count * args.max_first_per_member}. Increase the limit or add more members.",
                file=sys.stderr,
            )
            return 1
        if args.max_second_per_member is not None and member_count * args.max_second_per_member < proposal_count:
            print(
                f"Insufficient second-reviewer capacity: need {proposal_count} slots for {proposal_count} proposals, "
                f"but --max-second-per-member={args.max_second_per_member} with {member_count} members allows only "
                f"{member_count * args.max_second_per_member}. Increase the limit or add more members.",
                file=sys.stderr,
            )
            return 1
        if args.max_per_member is not None:
            total_required = proposal_count * args.reviewers_per_proposal
            total_capacity = member_count * args.max_per_member
            if total_capacity < total_required:
                print(
                    f"Insufficient reviewer capacity: assignments require {total_required} slots "
                    f"({proposal_count} proposals × {args.reviewers_per_proposal} reviewers), "
                    f"but --max-per-member={args.max_per_member} with {member_count} members allows only {total_capacity}.",
                    file=sys.stderr,
                )
                return 1
        try:
            role_labels = generate_role_labels(args.reviewers_per_proposal)
            member_assignments = assign_reviewers(
                proposals,
                members,
                args.reviewers_per_proposal,
                args.max_per_member,
                fixed_preferences,
                args.max_first_per_member,
                args.max_second_per_member,
                chair_members,
                manual_conflicts or None,
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

        assignments_path = args.assignments or Path("reviewer_assignments.txt")
        try:
            write_assignments(proposals, assignments_path, role_labels or [])
        except OSError as exc:
            print(f"Failed to write assignments: {exc}", file=sys.stderr)
            return 1
        if args.member_summary and member_assignments is not None:
            try:
                write_member_summary(member_assignments, args.member_summary, role_labels or [])
            except OSError as exc:
                print(f"Failed to write member summary: {exc}", file=sys.stderr)
                return 1

    output_lines: List[str] = []

    for proposal in proposals:
        proposal.pop("normalised_text", None)
        proposal.pop("participants", None)
        rendered = list(
            render_record(
                proposal["exp"],
                proposal["pi"].title(),
                proposal["nets"],
                proposal["lambda"],
                proposal["title"],
                proposal.get("first_reviewer"),
                proposal.get("second_reviewer"),
            )
        )
        output_lines.extend(rendered)
        output_lines.append("")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(output_lines)
        if output_lines and not content.endswith("\n"):
            content += "\n"
        args.output.write_text(content, encoding="utf-8")
    else:
        for line in output_lines:
            print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
