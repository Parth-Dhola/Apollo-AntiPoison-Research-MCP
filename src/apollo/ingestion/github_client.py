"""github_client.py — GitHub REST API client for open-source code and repo implementations."""

import logging
from typing import List, Optional

from apollo.config import get_settings
from apollo.models.schemas import CodeSnippetResult
from apollo.utils.cache import get_cache

logger = logging.getLogger("apollo.github")

_GITHUB_REPO_SEARCH_URL = "https://api.github.com/search/repositories"


async def search_github_repos(
    topic: str,
    language: Optional[str] = None,
    min_stars: int = 10,
    per_page: int = 5
) -> List[CodeSnippetResult]:
    """
    Search GitHub for repositories and key implementations.
    Uses free public rate limits (or GITHUB_TOKEN if configured).
    """
    if not topic:
        return []

    try:
        import httpx
    except ImportError:
        logger.error("httpx is not installed.")
        return []

    settings = get_settings()
    cache = get_cache()
    cache_key = {
        "topic": topic,
        "language": language,
        "min_stars": min_stars,
        "per_page": per_page
    }

    cached = cache.get("github_search", cache_key)
    if cached:
        logger.debug("Returning cached GitHub search results.")
        return [CodeSnippetResult(**item) for item in cached]

    # Build search query string
    q_parts = [topic]
    if language:
        q_parts.append(f"language:{language}")
    if min_stars > 0:
        q_parts.append(f"stars:>={min_stars}")

    query_str = " ".join(q_parts)
    params = {
        "q": query_str,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page
    }

    headers = {
        "User-Agent": settings.USER_AGENT,
        "Accept": "application/vnd.github.v3+json"
    }
    if settings.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {settings.GITHUB_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(_GITHUB_REPO_SEARCH_URL, params=params, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"GitHub API returned status {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            items = data.get("items", [])

            results: List[CodeSnippetResult] = []
            for repo in items:
                repo_name = repo.get("full_name") or "unknown"
                repo_url = repo.get("html_url") or ""
                description = repo.get("description") or "(No description)"
                stars = repo.get("stargazers_count") or 0
                lang = repo.get("language")
                default_branch = repo.get("default_branch", "main")

                # Try to fetch repo README preview via raw content (free & zero rate limit)
                readme_content = f"# {repo_name}\n{description}\nStars: {stars}\n"
                try:
                    readme_url = f"https://raw.githubusercontent.com/{repo_name}/{default_branch}/README.md"
                    r_resp = await client.get(readme_url, timeout=3.0)
                    if r_resp.status_code == 200 and r_resp.text:
                        # Extract first 1000 characters of README
                        readme_preview = r_resp.text[:1000]
                        readme_content = f"Repository: {repo_name} (⭐ {stars})\n\n{readme_preview}"
                except Exception:
                    pass

                snippet_res = CodeSnippetResult(
                    repo_name=repo_name,
                    repo_url=repo_url,
                    file_path="README.md",
                    language=lang,
                    stars=stars,
                    snippet=readme_content,
                    description=description
                )
                results.append(snippet_res)

            cache.set("github_search", cache_key, [r.model_dump() for r in results])
            return results

    except Exception as e:
        logger.error(f"Failed to query GitHub API: {e}")
        return []

