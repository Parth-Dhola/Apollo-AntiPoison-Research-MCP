"""normalizer.py — Normalizes LaTeX math equations and cleans Markdown syntax."""

import re


def normalize_latex_and_markdown(text: str) -> str:
    """
    Standardize LaTeX equations and cleanup Markdown structures for consistent rendering
    across Aurora, Obsidian, and LLM reasoning prompts.
    """
    if not text:
        return ""

    # 1. Convert display math blocks \[ ... \] -> $$ ... $$
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\n\1\n$$", text, flags=re.DOTALL)

    # 2. Convert inline math \( ... \) -> $ ... $
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text)

    # 3. Standardize LaTeX environments (equation, align, gather) to block math
    text = re.sub(
        r"\\begin\{(?:equation|align|gather|aligned)\*?\}(.*?)\\end\{(?:equation|align|gather|aligned)\*?\}",
        r"$$\n\1\n$$",
        text,
        flags=re.DOTALL
    )

    # 4. Fix excessive newlines (max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Fix corrupted markdown headers like ####### (standardize max 4 levels if deeper)
    text = re.sub(r"^#{5,}\s+", "#### ", text, flags=re.MULTILINE)

    # 6. Trim leading/trailing whitespace on lines
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()

