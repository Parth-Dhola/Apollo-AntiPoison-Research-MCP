"""snippet_packer.py — Packs and formats reranked snippets into dense grounded context blocks with token budgeting."""

from typing import List
from apollo.models.schemas import GroundedContextSnippet
from apollo.sanitization.anti_poison import wrap_in_secure_xml, sanitize_untrusted_text
from apollo.sanitization.normalizer import normalize_latex_and_markdown
from apollo.sanitization.noise_reducer import remove_academic_noise, clean_code_boilerplate


def pack_grounded_snippets(
    snippets: List[GroundedContextSnippet],
    max_length_per_snippet: int = 1000,
    min_relevance_score: float = 0.005,
    max_total_chars: int = 2500
) -> str:
    """
    Format and pack top-K snippets into an anti-poisoned, dense context string.
    Enforces strict relevance score thresholding and token budgeting to prevent context overload.
    """
    if not snippets:
        return "(No verified academic or code context found.)"

    # Filter out low-relevance snippets to prevent noise pollution
    filtered_snippets = [s for s in snippets if s.relevance_score >= min_relevance_score]
    if not filtered_snippets:
        # If all were below threshold, take only the highest scoring one
        filtered_snippets = [snippets[0]]

    blocks = []
    total_chars = 0
    seen_titles = set()

    for idx, snippet in enumerate(filtered_snippets, start=1):
        # Deduplication
        normalized_title = snippet.title.lower().strip()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)

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

        # Authority Tier Tag
        authority_badges = {
            "arxiv": "🏛️ [Tier 1: Peer-Reviewed / Primary Academic Literature]",
            "semantic_scholar": "🏛️ [Tier 1: Peer-Reviewed / Primary Academic Literature]",
            "github": "💻 [Tier 2: Verified Open-Source Code Implementation]",
            "wikipedia": "📖 [Tier 3: Crowd-Sourced Encyclopedia — Secondary / Conceptual Reference]",
            "web": "🌐 [Tier 4: General Web Search Context]"
        }
        authority_tag = authority_badges.get(snippet.source, "General Context")

        meta_line = " | ".join(meta_items) if meta_items else "General Context"
        url_line = f"URL: {snippet.url}" if snippet.url else ""

        status_tag = "⚠️ [Sanitized Injection Redacted]" if was_flagged else "✓ [Verified Safe]"

        block = (
            f"### Snippet {idx}: {snippet.title} ({snippet.source.upper()})\n"
            f"- **Source Authority**: {authority_tag}\n"
            f"- **Metadata**: {meta_line}\n"
            f"- **Status**: {status_tag} | Weighted Relevance: {snippet.relevance_score:.3f}\n"
            f"{f'- **Link**: {url_line}' if url_line else ''}\n\n"
            f"{wrap_in_secure_xml(sanitized_content, source=snippet.source, identifier=snippet.title)}"
        )

        if total_chars + len(block) > max_total_chars and blocks:
            # Respect token budget
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(blocks)
