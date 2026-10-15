"""Section-aware chunking of plan booklets.

One chunk per `§` section that has its own body text. The chunk ID is the
citation ID ("PLAN-A §4.2.1"), so a retrieved chunk can be cited without any
lookup. Each chunk's indexed text starts with the plan name, its own heading
and its parent headings ("Part of: §4 Health benefits > §4.2 Paramedical
services"). That way a short leaf section such as "Physiotherapy" still
matches queries about "paramedical" or "Bronze".

Tables are flattened to "label: value" lines for indexing. The original
Markdown is kept for display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from benefitpilot.config import PLAN_DOCS_DIR

TITLE_RE = re.compile(r"^#\s+(PLAN-[A-Z0-9]+)\s+—\s+(\w+)")
SECTION_RE = re.compile(r"^#{1,6}\s+§([\d.]+)\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str  # citation ID, e.g. "PLAN-A §4.2.1"
    plan_id: str
    plan_name: str
    section: str
    title: str
    parents: tuple[str, ...]  # e.g. ("§4 Health benefits", "§4.2 Paramedical services")
    text: str  # what gets indexed
    markdown: str  # original body, for display


def _parent_sections(section: str) -> list[str]:
    parts = section.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _flatten_tables(lines: list[str]) -> list[str]:
    out: list[str] = []
    header: list[str] | None = None
    for line in lines:
        if not line.startswith("|"):
            header = None
            out.append(line)
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        if header is None:
            header = cells
            continue
        if header[:2] == ["Rule", "Value"]:
            out.append(f"{cells[0]}: {cells[1]}")
        else:
            out.append(f"{cells[0]} — {header[1]}: {cells[1]}")
    return out


def chunk_markdown(markdown: str) -> list[Chunk]:
    lines = markdown.splitlines()
    m = TITLE_RE.match(lines[0]) if lines else None
    if m is None:
        raise ValueError("booklet must start with '# PLAN-X — Name ...'")
    plan_id, plan_name = m.groups()

    sections: list[tuple[str, str, list[str]]] = []  # (number, title, body lines)
    for line in lines[1:]:
        if s := SECTION_RE.match(line):
            sections.append((s.group(1), s.group(2), []))
        elif sections:
            sections[-1][2].append(line)
        # Lines before §1 (the synthetic-data notice) are not indexed.

    titles = {num: title for num, title, _ in sections}
    chunks = []
    for num, title, body in sections:
        body_md = "\n".join(body).strip()
        if not body_md:
            continue  # heading-only parents are carried into children's context
        parents = tuple(f"§{p} {titles[p]}" for p in _parent_sections(num) if p in titles)
        header = f"{plan_id} ({plan_name} plan) — §{num} {title}"
        if parents:
            header += "\nPart of: " + " > ".join(parents)
        text = header + "\n\n" + "\n".join(_flatten_tables(body)).strip()
        chunks.append(
            Chunk(
                id=f"{plan_id} §{num}",
                plan_id=plan_id,
                plan_name=plan_name,
                section=num,
                title=title,
                parents=parents,
                text=text,
                markdown=body_md,
            )
        )
    return chunks


def load_chunks(directory: Path = PLAN_DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8")))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk IDs")
    return chunks
