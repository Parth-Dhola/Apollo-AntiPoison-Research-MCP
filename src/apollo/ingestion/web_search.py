"""web_search.py — Resilient DuckDuckGo web search fallback (100% free, zero keys)."""

import logging
from typing import List, Dict, Any
from apollo.utils.cache import get_cache

logger = logging.getLogger("apollo.web_search")


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Execute a free DuckDuckGo search query."""
    if not query:
        return []

    cache = get_cache()
    cache_key = {"query": query, "max_results": max_results}
    cached = cache.get("ddg_search", cache_key)
    if cached:
        return cached

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("link", ""),
                "content": r.get("body", "")
            }
            for r in results
        ]
        cache.set("ddg_search", cache_key, formatted, ttl=3600)  # 1 hour cache
        return formatted

    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return []

