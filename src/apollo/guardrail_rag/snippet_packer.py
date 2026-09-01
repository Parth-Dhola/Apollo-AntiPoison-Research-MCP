"""snippet_packer.py — Packs and formats reranked snippets into dense grounded context blocks."""

from typing import List
from apollo.models.schemas import GroundedContextSnippet
from apollo.sanitization.anti_poison import wrap_in_secure_xml, sanitize_untrusted_text
from apollo.sanitization.normalizer import normalize_latex_and_markdown
from apollo.sanitization.noise_reducer import remove_academic_noise, clean_code_boilerplate


def pack_grounded_snippets(
    snippets: List[GroundedContextSnippet],
    max_length_per_snippet: int = 1200
) -> str:
    """
    Format and pack top-K snippets into an anti-poisoned, dense context string
    suitable for LLM reasoning and Obsidian notes.
    """
    if not snippets:
        return "(No verified academic or code context found.)"

    blocks = []
    for idx, snippet in enumerate(snippets, start=1):
        # 1. Strip noise
        content = remove_academic_noise(snippet.content)
        if snippet.source == "github":
            content = clean_code_boilerplate(content)

        # 2. Normalize equations & markdown
        content = normalize_latex_and_markdown(content)

        # 3. Anti-poisoning & length cap
        sanitized_content, was_flagged = sanitize_untrusted_text(content, max_length=max_length_per_snippet)

        # 4. Build citation header
        meta = snippet.citation_meta
        meta_items = []
        if "arxiv_id" in meta:
            meta_items.append(f"arXiv: {meta['arxiv_id']}")
        if "year" in meta and meta["year"]:
            meta_items.append(f"Year: {meta['year']}")
        if "citations" in meta and meta["citations"] is not None:
            meta_items.append(f"Citations: {meta['citations']}")
        if "stars" in meta:
            meta_items.append(f"Stars: ⭐ {meta['stars']}")
        if "language" in meta and meta["language"]:
            meta_items.append(f"Lang: {meta['language']}")
        if "authors" in meta and meta["authors"]:
            authors = meta["authors"]
            if isinstance(authors, list):
                author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            else:
                author_str = str(authors)
            meta_items.append(f"Authors: {author_str}")

        meta_line = " | ".join(meta_items) if meta_items else "General Context"
        url_line = f"URL: {snippet.url}" if snippet.url else ""

        status_tag = "⚠️ [Sanitized Injection Redacted]" if was_flagged else "✓ [Verified Safe]"

        block = (
            f"### Snippet {idx}: {snippet.title} ({snippet.source.upper()})\n"
            f"- **Metadata**: {meta_line}\n"
            f"- **Status**: {status_tag} | Relevance: {snippet.relevance_score:.3f}\n"
            f"{f'- **Link**: {url_line}' if url_line else ''}\n\n"
            f"{wrap_in_secure_xml(sanitized_content, source=snippet.source, identifier=snippet.title)}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)

