"""anti_poison.py — Prompt injection scanner, invisible unicode stripper, and secure XML wrapper."""

import re
import html
import unicodedata
from typing import Tuple, List

# Common prompt injection triggers and delimiter break-outs
_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|directions)\b",
    r"(?i)\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\b",
    r"(?i)\byou\s+are\s+now\s+(?:acting\s+as|an\s+unfiltered|in\s+DAN\s+mode|a\s+different\s+model)\b",
    r"(?i)\bsystem\s+override\b",
    r"(?i)\bnew\s+system\s+(?:prompt|directive|rule)\b",
    r"(?i)\boutput\s+(?:your\s+)?(?:entire|initial|system)\s+prompt\b",
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
    r"(?i)\[INST\]",
    r"(?i)\[/INST\]",
    r"(?i)<<SYS>>",
    r"(?i)<</SYS>>",
    r"(?i)\bhuman:\s*",
    r"(?i)\bassistant:\s*",
]

_COMPILED_INJECTION_RE = [re.compile(p) for p in _INJECTION_PATTERNS]

# Invisible Unicode / BiDi override characters
_INVISIBLE_UNICODE_CHARS = [
    "\u200B",  # Zero-width space
    "\u200C",  # Zero-width non-joiner
    "\u200D",  # Zero-width joiner
    "\uFEFF",  # Byte order mark / zero-width no-break space
    "\u202A",  # Left-to-right embedding
    "\u202B",  # Right-to-left embedding
    "\u202C",  # Pop directional formatting
    "\u202D",  # Left-to-right override
    "\u202E",  # Right-to-left override
    "\u2060",  # Word joiner
    "\u2066",  # Left-to-right isolate
    "\u2067",  # Right-to-left isolate
    "\u2068",  # First strong isolate
    "\u2069",  # Pop directional isolate
]


def strip_invisible_unicode(text: str) -> str:
    """Remove hidden zero-width, bidirectional overrides, and non-printable control characters."""
    if not text:
        return ""
    # Normalize unicode to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Strip specific known invisible/bidi chars
    for char in _INVISIBLE_UNICODE_CHARS:
        text = text.replace(char, "")
    # Remove other non-printable chars (excluding standard newlines, tabs)
    cleaned_chars = []
    for c in text:
        if c in ("\n", "\r", "\t") or unicodedata.category(c) not in ("Cc", "Cf", "Co", "Cs"):
            cleaned_chars.append(c)
    return "".join(cleaned_chars)


def scan_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """
    Scan text for prompt injection patterns and adversarial prompt override attempts.
    Returns (is_suspicious: bool, detected_patterns: List[str]).
    """
    if not text:
        return False, []

    matched = []
    for regex in _COMPILED_INJECTION_RE:
        matches = regex.findall(text)
        if matches:
            matched.append(regex.pattern)

    return len(matched) > 0, matched


def sanitize_untrusted_text(text: str, max_length: int = 1500) -> Tuple[str, bool]:
    """
    Sanitize untrusted text retrieved from external sources.
    1. Strips invisible unicode
    2. Redacts dangerous prompt injection sequences
    3. Caps length to prevent token flood attacks
    Returns (sanitized_text: str, injection_neutralized: bool)
    """
    if not text:
        return "", False

    text = strip_invisible_unicode(text)
    is_suspicious, patterns = scan_prompt_injection(text)

    if is_suspicious:
        # Redact matches instead of completely discarding potentially useful academic content
        for regex in _COMPILED_INJECTION_RE:
            text = regex.sub("[REDACTED_ADVERSARIAL_INSTRUCTION]", text)

    # Length capping
    if len(text) > max_length:
        text = text[:max_length] + " ... [TRUNCATED]"

    return text.strip(), is_suspicious


def wrap_in_secure_xml(content: str, source: str, identifier: str = "") -> str:
    """
    Wrap untrusted content in hardened XML delimiters so LLMs treat it strictly as data,
    not instructions.
    """
    clean_id = html.escape(identifier)
    clean_src = html.escape(source)
    return f'<untrusted_academic_context source="{clean_src}" id="{clean_id}">\n{content}\n</untrusted_academic_context>'

