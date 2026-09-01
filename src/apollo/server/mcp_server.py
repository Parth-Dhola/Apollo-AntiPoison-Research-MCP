"""mcp_server.py — Standalone FastMCP Server exposing clean, anti-poisoned research tools."""

import asyncio
import logging
import time
from typing import Optional, List, Tuple
from fastmcp import FastMCP

from apollo.config import get_settings
from apollo.ingestion.arxiv_client import search_arxiv, fetch_arxiv_paper
from apollo.ingestion.semantic_scholar import search_semantic_scholar
from apollo.ingestion.github_client import search_github_repos
from apollo.ingestion.web_search import search_duckduckgo
from apollo.guardrail_rag.reranker import rank_snippets
from apollo.guardrail_rag.snippet_packer import pack_grounded_snippets
from apollo.models.schemas import (
    GroundedContextSnippet,
    QueryIntent,
    ResearchContextResult
)
from apollo.router.tool_selector import select_tools_for_query
from apollo.sanitization.anti_poison import sanitize_untrusted_text, wrap_in_secure_xml
from apollo.sanitization.normalizer import normalize_latex_and_markdown

logger = logging.getLogger("apollo.mcp")


def create_mcp_server() -> FastMCP:
    """Instantiate and configure the Apollo FastMCP Server."""
    mcp = FastMCP("apollo-research-server")

    # ── Tool 1: Academic Papers Search ─────────────────────────────────────────
    @mcp.tool()
    async def search_academic_papers(
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: int = 0,
        top_k: int = 5
    ) -> str:
        """
        Search peer-reviewed papers and preprints across arXiv and Semantic Scholar.
        Returns sanitized paper abstracts with publication years, citation counts, and authors.
        """
        year_range = (year_start, year_end) if year_start and year_end else None

        # Fetch in parallel from arXiv and Semantic Scholar
        arxiv_task = search_arxiv(query, max_results=top_k)
        s2_task = search_semantic_scholar(query, limit=top_k, min_citations=min_citations, year_range=year_range)

        arxiv_papers, s2_papers = await asyncio.gather(arxiv_task, s2_task)

        # Convert to GroundedContextSnippet
        candidate_snippets: List[GroundedContextSnippet] = []

        for p in arxiv_papers:
            candidate_snippets.append(GroundedContextSnippet(
                source="arxiv",
                title=p.title,
                url=p.url or (f"https://arxiv.org/abs/{p.arxiv_id}" if p.arxiv_id else None),
                content=p.abstract,
                citation_meta={
                    "arxiv_id": p.arxiv_id,
                    "year": p.year,
                    "authors": p.authors
                }
            ))

        for p in s2_papers:
            candidate_snippets.append(GroundedContextSnippet(
                source="semantic_scholar",
                title=p.title,
                url=p.url,
                content=p.abstract,
                citation_meta={
                    "doi": p.doi,
                    "year": p.year,
                    "citations": p.citation_count,
                    "authors": p.authors
                }
            ))

        if not candidate_snippets:
            return "No academic papers found matching the query."

        # Rerank with FlashRank/BM25
        ranked = rank_snippets(query, candidate_snippets, top_k=top_k)
        return pack_grounded_snippets(ranked)

    # ── Tool 2: Deep Paper Context & Equations ────────────────────────────────
    @mcp.tool()
    async def fetch_paper_deep_context(
        arxiv_id: str,
        max_tokens: int = 1500
    ) -> str:
        """
        Fetch structured, sanitized abstract and detailed technical context for a specific arXiv ID.
        Normalizes LaTeX equations and neutralizes any prompt injection payloads.
        """
        paper = await fetch_arxiv_paper(arxiv_id)
        if not paper:
            return f"Could not find arXiv paper with ID: {arxiv_id}"

        clean_abstract, was_flagged = sanitize_untrusted_text(paper.abstract, max_length=max_tokens)
        clean_abstract = normalize_latex_and_markdown(clean_abstract)

        authors_str = ", ".join(paper.authors[:4]) + (" et al." if len(paper.authors) > 4 else "")
        categories_str = ", ".join(paper.categories) if paper.categories else "cs.AI"

        header = (
            f"# {paper.title}\n"
            f"- **arXiv ID**: {paper.arxiv_id}\n"
            f"- **Authors**: {authors_str}\n"
            f"- **Year**: {paper.year or 'Recent'}\n"
            f"- **Categories**: {categories_str}\n"
            f"- **PDF Link**: {paper.pdf_url or f'https://arxiv.org/pdf/{paper.arxiv_id}'}\n"
            f"- **Safety Status**: {'⚠️ [Flagged & Sanitized]' if was_flagged else '✓ [Verified Safe]'}\n\n"
        )

        xml_body = wrap_in_secure_xml(clean_abstract, source="arxiv", identifier=paper.arxiv_id)
        return header + xml_body

    # ── Tool 3: Code Implementations & Repos ───────────────────────────────────
    @mcp.tool()
    async def search_repo_implementations(
        topic: str,
        language: Optional[str] = "python",
        min_stars: int = 20,
        top_k: int = 3
    ) -> str:
        """
        Search verified open-source GitHub repositories for code implementations and architectural patterns.
        Strips license boilerplate and formats clean implementation snippets.
        """
        repos = await search_github_repos(topic=topic, language=language, min_stars=min_stars, per_page=top_k * 2)
        if not repos:
            return f"No open-source repositories found matching topic: {topic}"

        candidate_snippets = [
            GroundedContextSnippet(
                source="github",
                title=f"{r.repo_name} ({r.file_path})",
                url=r.repo_url,
                content=r.snippet,
                citation_meta={
                    "stars": r.stars,
                    "language": r.language or language
                }
            )
            for r in repos
        ]

        ranked = rank_snippets(topic, candidate_snippets, top_k=top_k)
        return pack_grounded_snippets(ranked)

    # ── Tool 4: DuckDuckGo Fallback Search ─────────────────────────────────────
    @mcp.tool()
    async def fallback_web_search(
        query: str,
        max_results: int = 3
    ) -> str:
        """
        Zero-cost DuckDuckGo search fallback for recent news, product launches, and general topics.
        Applies anti-poisoning filter to all retrieved snippets.
        """
        results = search_duckduckgo(query, max_results=max_results)
        if not results:
            return f"No web search results found for: {query}"

        candidate_snippets = [
            GroundedContextSnippet(
                source="web",
                title=r["title"],
                url=r["url"],
                content=r["content"],
                citation_meta={"engine": "duckduckgo"}
            )
            for r in results
        ]

        ranked = rank_snippets(query, candidate_snippets, top_k=max_results)
        return pack_grounded_snippets(ranked)

    # ── Tool 5: Unified Context Engine (Flagship Router + Filter) ──────────────
    @mcp.tool()
    async def unified_research_context(
        query: str,
        top_k: int = 3
    ) -> str:
        """
        End-to-end multi-source research context pipeline.
        1. Classifies intent using local zero-cost Bag-of-Words router.
        2. Aggregates candidates from arXiv, Semantic Scholar, GitHub, and Web.
        3. Filters prompt injections and normalizes LaTeX math.
        4. Reranks with FlashRank/BM25 and returns dense grounded context.
        """
        start_time = time.time()
        selection = select_tools_for_query(query)

        tasks = []
        if selection.intent in (QueryIntent.ACADEMIC_PAPER, QueryIntent.DEEP_THEORY, QueryIntent.HYBRID):
            tasks.append(search_arxiv(query, max_results=top_k))
            tasks.append(search_semantic_scholar(query, limit=top_k))

        if selection.intent in (QueryIntent.CODE_IMPLEMENTATION, QueryIntent.HYBRID):
            tasks.append(search_github_repos(topic=query, per_page=top_k))

        if selection.intent == QueryIntent.GENERAL_WEB or not tasks:
            # Fallback web search
            ddg_results = search_duckduckgo(query, max_results=top_k)
            candidates = [
                GroundedContextSnippet(
                    source="web",
                    title=r["title"],
                    url=r["url"],
                    content=r["content"],
                    citation_meta={"engine": "duckduckgo"}
                )
                for r in ddg_results
            ]
            ranked = rank_snippets(query, candidates, top_k=top_k)
            return pack_grounded_snippets(ranked)

        # Execute parallel ingestion
        ingestion_results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: List[GroundedContextSnippet] = []
        for res in ingestion_results:
            if isinstance(res, Exception) or not res:
                continue

            for item in res:
                if hasattr(item, "abstract"):  # PaperMetadata
                    candidates.append(GroundedContextSnippet(
                        source=item.source,
                        title=item.title,
                        url=item.url or (f"https://arxiv.org/abs/{item.arxiv_id}" if item.arxiv_id else None),
                        content=item.abstract,
                        citation_meta={
                            "arxiv_id": item.arxiv_id,
                            "doi": item.doi,
                            "year": item.year,
                            "citations": item.citation_count,
                            "authors": item.authors
                        }
                    ))
                elif hasattr(item, "repo_name"):  # CodeSnippetResult
                    candidates.append(GroundedContextSnippet(
                        source="github",
                        title=f"{item.repo_name} ({item.file_path})",
                        url=item.repo_url,
                        content=item.snippet,
                        citation_meta={
                            "stars": item.stars,
                            "language": item.language
                        }
                    ))

        if not candidates:
            # Fallback to web search
            ddg_results = search_duckduckgo(query, max_results=top_k)
            candidates = [
                GroundedContextSnippet(
                    source="web",
                    title=r["title"],
                    url=r["url"],
                    content=r["content"],
                    citation_meta={"engine": "duckduckgo"}
                )
                for r in ddg_results
            ]

        # Rerank with FlashRank/BM25
        ranked = rank_snippets(query, candidates, top_k=top_k)
        packed_text = pack_grounded_snippets(ranked)

        latency_ms = (time.time() - start_time) * 1000
        summary_header = (
            f"<!-- Apollo Context Engine | Intent: {selection.intent.value} "
            f"| Confidence: {selection.confidence * 100:.0f}% | Latency: {latency_ms:.1f}ms -->\n\n"
        )

        return summary_header + packed_text

    return mcp

