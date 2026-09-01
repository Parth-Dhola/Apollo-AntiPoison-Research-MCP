"""Sanitization and Anti-Poisoning Layer."""

from apollo.sanitization.anti_poison import (
    sanitize_untrusted_text,
    scan_prompt_injection,
    strip_invisible_unicode,
    wrap_in_secure_xml
)
from apollo.sanitization.normalizer import normalize_latex_and_markdown
from apollo.sanitization.noise_reducer import remove_academic_noise, clean_code_boilerplate

__all__ = [
    "sanitize_untrusted_text",
    "scan_prompt_injection",
    "strip_invisible_unicode",
    "wrap_in_secure_xml",
    "normalize_latex_and_markdown",
    "remove_academic_noise",
    "clean_code_boilerplate"
]

