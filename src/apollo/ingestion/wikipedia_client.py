"""wikipedia_client.py — Free Wikipedia API client for foundational concepts & overviews."""

import logging
import urllib.parse
from typing import List, Dict, Any, Optional

from apollo.config import get_settings
from apollo.utils.cache import get_cache

logger = logging.getLogger("apollo.wikipedia")

_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"


async def search_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search Wikipedia for foundational knowledge, algorithms, and definitions.
    100% Free, zero API keys required, generous rate limits.
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
    cache_key = {"query": query, "max_results": max_results}

    cached = cache.get("wiki_search", cache_key)
    if cached:
        logger.debug("Returning cached Wikipedia search results.")
        return cached

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results,
        "utf8": 1
    }

    headers = {"User-Agent": settings.USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(_WIKI_SEARCH_URL, params=search_params, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"Wikipedia search API returned status {resp.status_code}")
                return []

            data = resp.json()
            search_items = data.get("query", {}).get("search", [])

            results: List[Dict[str, Any]] = []

            for item in search_items:
                title = item.get("title", "")
                if not title:
                    continue

                # Fetch structured page extract/summary
                encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                summary_url = f"{_WIKI_SUMMARY_URL}/{encoded_title}"

                try:
                    s_resp = await client.get(summary_url, headers=headers, timeout=5.0)
                    if s_resp.status_code == 200:
                        s_data = s_resp.json()
                        extract = s_data.get("extract") or ""
                        page_url = s_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded_title}")
                        description = s_data.get("description") or ""

                        if extract:
                            results.append({
                                "title": title,
                                "url": page_url,
                                "description": description,
                                "content": extract,
                                "source": "wikipedia"
                            })
                            continue
                except Exception as e:
                    logger.debug(f"Could not fetch summary for {title}: {e}")

                # Fallback to snippet from search API if summary endpoint didn't respond
                import re
                raw_snippet = item.get("snippet", "")
                clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                results.append({
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{encoded_title}",
                    "description": "",
                    "content": clean_snippet,
                    "source": "wikipedia"
                })

            cache.set("wiki_search", cache_key, results, ttl=86400)  # 24 hour cache
            return results

    except Exception as e:
        logger.error(f"Failed to query Wikipedia API: {e}")
        return []


async def fetch_wikipedia_article(title: str) -> Optional[Dict[str, Any]]:
    """Fetch complete summary extract for a specific Wikipedia article title."""
    if not title:
        return None

    try:
        import httpx
    except ImportError:
        return None

    settings = get_settings()
    cache = get_cache()
    cached = cache.get("wiki_article", title)
    if cached:
        return cached

    encoded_title = urllib.parse.quote(title.replace(" ", "_"))
    summary_url = f"{_WIKI_SUMMARY_URL}/{encoded_title}"
    headers = {"User-Agent": settings.USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(summary_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                result = {
                    "title": data.get("title", title),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded_title}"),
                    "description": data.get("description", ""),
                    "content": data.get("extract", ""),
                    "source": "wikipedia"
                }
                cache.set("wiki_article", title, result, ttl=86400)
                return result
    except Exception as e:
        logger.error(f"Failed to fetch Wikipedia article {title}: {e}")

    return None
