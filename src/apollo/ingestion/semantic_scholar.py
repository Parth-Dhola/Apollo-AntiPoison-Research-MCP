"""semantic_scholar.py — Semantic Scholar public Graph API client with free-tier support."""

import logging
from typing import List, Optional

from apollo.config import get_settings
from apollo.models.schemas import PaperMetadata
from apollo.utils.cache import get_cache

logger = logging.getLogger("apollo.semanticscholar")

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


async def search_semantic_scholar(
    query: str,
    limit: int = 5,
    min_citations: int = 0,
    year_range: Optional[tuple] = None
) -> List[PaperMetadata]:
    """
    Search Semantic Scholar Graph API for peer-reviewed papers.
    Uses free public tier with optional API key header.
    """
    if not query:
        return []

    try:
        import httpx
    except ImportError:
        logger.error("httpx is not installed.")
        return []

    settings = get_settings()
    cache = get_cache()
    cache_key = {
        "query": query,
        "limit": limit,
        "min_citations": min_citations,
        "year_range": year_range
    }

    cached = cache.get("s2_search", cache_key)
    if cached:
        logger.debug("Returning cached Semantic Scholar results.")
        return [PaperMetadata(**item) for item in cached]

    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,abstract,year,citationCount,authors,url,fieldsOfStudy,externalIds"
    }

    if year_range and len(year_range) == 2:
        params["year"] = f"{year_range[0]}-{year_range[1]}"

    headers = {"User-Agent": settings.USER_AGENT}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(_S2_SEARCH_URL, params=params, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"Semantic Scholar API status {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            papers_data = data.get("data", [])

            results: List[PaperMetadata] = []
            for item in papers_data:
                citations = item.get("citationCount") or 0
                if citations < min_citations:
                    continue

                ext_ids = item.get("externalIds") or {}
                arxiv_id = ext_ids.get("ArXiv")
                doi = ext_ids.get("DOI")

                author_names = [a.get("name") for a in item.get("authors", []) if a.get("name")]

                fields = item.get("fieldsOfStudy") or []

                abstract = item.get("abstract") or "(Abstract not indexed)"

                paper = PaperMetadata(
                    arxiv_id=arxiv_id,
                    doi=doi,
                    title=item.get("title") or "Untitled Paper",
                    authors=author_names,
                    year=item.get("year"),
                    abstract=abstract,
                    citation_count=citations,
                    url=item.get("url"),
                    categories=fields,
                    source="semantic_scholar"
                )
                results.append(paper)

            cache.set("s2_search", cache_key, [p.model_dump() for p in results])
            return results

    except Exception as e:
        logger.error(f"Failed to query Semantic Scholar API: {e}")
        return []

