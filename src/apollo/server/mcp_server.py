"""mcp_server.py — Standalone FastMCP Server exposing clean, anti-poisoned research tools."""

import asyncio
import logging
import time
from typing import Optional, List, Tuple, Dict, Any
from fastmcp import FastMCP

from apollo.config import get_settings
from apollo.ingestion.arxiv_client import search_arxiv, fetch_arxiv_paper
from apollo.ingestion.semantic_scholar import search_semantic_scholar
from apollo.ingestion.github_client import search_github_repos
from apollo.ingestion.web_search import search_duckduckgo
from apollo.ingestion.wikipedia_client import search_wikipedia as search_wikipedia_fn, fetch_wikipedia_article
from apollo.guardrail_rag.reranker import rank_snippets
from apollo.guardrail_rag.snippet_packer import pack_grounded_snippets
from apollo.models.schemas import (
    GroundedContextSnippet,
    QueryIntent,
    ResearchContextResult
)
from apollo.router.tool_rag import rank_tools_for_query
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

        arxiv_task = search_arxiv(query, max_results=top_k)
        s2_task = search_semantic_scholar(query, limit=top_k, min_citations=min_citations, year_range=year_range)

        arxiv_papers, s2_papers = await asyncio.gather(arxiv_task, s2_task)

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

    # ── Tool 4: Wikipedia Encyclopedia Search ─────────────────────────────────
    @mcp.tool()
    async def search_wikipedia(
        query: str,
        max_results: int = 3
    ) -> str:
        """
        Search Wikipedia encyclopedia for foundational definitions, algorithms, mathematical concepts, and historical context.
        Generous rate limits, 100% free, ideal for reliable conceptual overviews.
        """
        results = await search_wikipedia_fn(query, max_results=max_results)
        if not results:
            return f"No Wikipedia articles found for: {query}"

        candidate_snippets = [
            GroundedContextSnippet(
                source="wikipedia",
                title=r["title"],
                url=r["url"],
                content=f"{r.get('description', '')}\n\n{r['content']}".strip(),
                citation_meta={"source": "wikipedia"}
            )
            for r in results
        ]

        ranked = rank_snippets(query, candidate_snippets, top_k=max_results)
        return pack_grounded_snippets(ranked)

    # ── Tool 5: DuckDuckGo Fallback Search ─────────────────────────────────────
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

    # ── Tool 6: Tool Capability RAG Inspector ──────────────────────────────────
    @mcp.tool()
    async def match_tools_for_query(
        query: str,
        max_tools: int = 2
    ) -> str:
        """
        RAG over Tool Capabilities: Evaluates semantic capability fit to pick ONLY necessary tools,
        pruning irrelevant tools to prevent token bloat and context overload.
        """
        selected_tools = rank_tools_for_query(query, max_tools=max_tools)
        lines = [f"# Tool Selection RAG for: \"{query}\""]
        for idx, t in enumerate(selected_tools, 1):
            lines.append(f"### {idx}. `{t['tool_name']}` (Score: {t['score']:.2f})")
            lines.append(f"- **Category**: {t.get('category', 'general')} | **Token Cost**: {t.get('token_cost', 'medium')}")
            lines.append(f"- **Selection Reason**: {t['reason']}\n")
        return "\n".join(lines)

    # ── Tool 7: Unified Context Engine with Adaptive Tool Budgeting ────────────
    @mcp.tool()
    async def unified_research_context(
        query: str,
        top_k: int = 2
    ) -> str:
        """
        End-to-end multi-source research context pipeline with Tool Capability RAG & Token Budgeting.
        1. RAG-selects ONLY the top 1-2 optimal tools based on semantic capability match.
        2. Prunes unneeded tools to prevent API waste and context overload.
        3. Sanitizes all retrieved snippets and removes prompt injection payloads.
        4. Cross-encoder reranks on CPU and returns bounded, dense grounded context.
        """
        start_time = time.time()
        
        # 1. RAG Tool Selection
        tool_matches = rank_tools_for_query(query, max_tools=2)
        selected_tool_names = [t["tool_name"] for t in tool_matches]

        tasks = []
        if "fetch_paper_deep_context" in selected_tool_names:
            # Exact arXiv ID short-circuit
            import re
            m = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", query)
            if m:
                deep_ctx = await fetch_paper_deep_context(m.group(0))
                return deep_ctx

        if "search_academic_papers" in selected_tool_names:
            tasks.append(search_arxiv(query, max_results=top_k))
            tasks.append(search_semantic_scholar(query, limit=top_k))

        if "search_repo_implementations" in selected_tool_names:
            tasks.append(search_github_repos(topic=query, per_page=top_k))

        if "search_wikipedia" in selected_tool_names:
            tasks.append(search_wikipedia_fn(query, max_results=top_k))

        if "fallback_web_search" in selected_tool_names or not tasks:
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

        # Execute parallel ingestion for ONLY selected tools
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
            # Graceful fallback to web search
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

        # Rerank with FlashRank/BM25 & enforce token budget
        ranked = rank_snippets(query, candidates, top_k=top_k)
        packed_text = pack_grounded_snippets(ranked, max_total_chars=2500)

        latency_ms = (time.time() - start_time) * 1000
        summary_header = (
            f"<!-- Apollo Context Engine | Tool RAG: {', '.join(selected_tool_names)} "
            f"| Latency: {latency_ms:.1f}ms | Budget: Bounded -->\n\n"
        )

        return summary_header + packed_text

    return mcp
