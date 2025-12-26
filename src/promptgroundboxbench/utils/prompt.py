from __future__ import annotations

import re


_SEP_RE = re.compile(r"[.;,\n]+")


def parse_prompt(prompt: str) -> list[str]:
    """Parse a user prompt into a list of label strings.

    Supports period separated prompts such as:
    "a person. a chair. a laptop."
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    parts = [p.strip() for p in _SEP_RE.split(prompt) if p.strip()]
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def normalize_label(label: str) -> str:
    """Normalize labels for consistent mapping and reporting."""
    s = label.strip().lower()
    s = s.removesuffix(".")
    s = re.sub(r"^(a|an|the)\s+", "", s).strip()
    return s
