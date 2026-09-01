"""noise_reducer.py — Strips academic noise (references, acknowledgments) and code boilerplates."""

import re

# Section headers indicating non-substantive sections
_NOISE_SECTION_PATTERNS = [
    r"(?i)\n#+\s*(?:References|Bibliography|Works Cited|Literature Cited)[\s\S]*$",
    r"(?i)\n#+\s*(?:Acknowledgements?|Funding|Conflict(?:s)? of Interest|Author Contributions)[\s\S]*?(?=\n#|\Z)",
    r"(?i)\b(?:Ethics Statement|Data Availability Statement)[\s\S]*?(?=\n#|\Z)"
]

_COMPILED_NOISE_SECTIONS = [re.compile(p) for p in _NOISE_SECTION_PATTERNS]

# Typical license boilerplate in code
_CODE_LICENSE_PATTERN = re.compile(
    r"(?s)^\s*(?:#|//|/\*|\*).*?(?:Copyright|Licensed under the Apache License|MIT License|BSD License|GNU General Public License).*?(?:\*/|\n\s*(?:import|from|class|def|package|use|include|\Z))",
    re.IGNORECASE
)


def remove_academic_noise(text: str) -> str:
    """Strip references, acknowledgments, and trailing academic boilerplate from text."""
    if not text:
        return ""

    cleaned = text
    for regex in _COMPILED_NOISE_SECTIONS:
        cleaned = regex.sub("", cleaned)

    return cleaned.strip()


def clean_code_boilerplate(code: str) -> str:
    """Remove large license/copyright comment blocks from top of code snippets."""
    if not code:
        return ""

    # Check for large comment blocks at top
    match = _CODE_LICENSE_PATTERN.match(code)
    if match:
        end_idx = match.end()
        # If match ends before code start, extract from the code keywords
        remaining = code[end_idx:].lstrip()
        if remaining:
            return remaining

    return code.strip()

