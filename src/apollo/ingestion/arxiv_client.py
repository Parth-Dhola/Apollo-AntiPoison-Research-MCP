"""arxiv_client.py — Free arXiv Atom API client with XML parsing and caching."""

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional

from apollo.config import get_settings
from apollo.models.schemas import PaperMetadata
from apollo.utils.cache import get_cache

logger = logging.getLogger("apollo.arxiv")

_ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _parse_arxiv_entry(entry: ET.Element) -> PaperMetadata:
    title = entry.findtext("atom:title", default="", namespaces=_ATOM_NS).strip().replace("\n", " ")
    abstract = entry.findtext("atom:summary", default="", namespaces=_ATOM_NS).strip().replace("\n", " ")
    id_url = entry.findtext("atom:id", default="", namespaces=_ATOM_NS).strip()
    
    # Extract arxiv_id from url (e.g. http://arxiv.org/abs/2307.08691v1)
    arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url

    published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS).strip()
    year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None

    authors = [
        author.findtext("atom:name", default="", namespaces=_ATOM_NS).strip()
        for author in entry.findall("atom:author", namespaces=_ATOM_NS)
    ]

    pdf_url = None
    for link in entry.findall("atom:link", namespaces=_ATOM_NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href")

    categories = [
        cat.get("term") for cat in entry.findall("atom:category", namespaces=_ATOM_NS)
        if cat.get("term")
    ]

    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        url=id_url,
        pdf_url=pdf_url,
        categories=categories,
        source="arxiv"
    )


async def search_arxiv(query: str, max_results: int = 5) -> List[PaperMetadata]:
    """Search arXiv public API with caching and error tolerance."""
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
    cached = cache.get("arxiv_search", cache_key)
    if cached:
        logger.debug("Returning cached arXiv search results.")
        return [PaperMetadata(**item) for item in cached]

    clean_query = urllib.parse.quote(query.strip())
    url = f"{_ARXIV_BASE_URL}?search_query=all:{clean_query}&start=0&max_results={max_results}&sortBy=relevance"

    headers = {"User-Agent": settings.USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.error(f"arXiv API error status {resp.status_code}: {resp.text[:200]}")
                return []

            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", namespaces=_ATOM_NS)
            papers = [_parse_arxiv_entry(e) for e in entries]

            # Cache the parsed results
            cache.set("arxiv_search", cache_key, [p.model_dump() for p in papers])
            return papers

    except Exception as e:
        logger.error(f"Failed to query arXiv API: {e}")
        return []


async def fetch_arxiv_paper(arxiv_id: str) -> Optional[PaperMetadata]:
    """Fetch metadata for a specific arXiv ID."""
    if not arxiv_id:
        return None

    try:
        import httpx
    except ImportError:
        return None

    cache = get_cache()
    cached = cache.get("arxiv_paper", arxiv_id)
    if cached:
        return PaperMetadata(**cached)

    clean_id = urllib.parse.quote(arxiv_id.strip())
    url = f"{_ARXIV_BASE_URL}?id_list={clean_id}&max_results=1"
    headers = {"User-Agent": get_settings().USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", namespaces=_ATOM_NS)
                if entries:
                    paper = _parse_arxiv_entry(entries[0])
                    cache.set("arxiv_paper", arxiv_id, paper.model_dump())
                    return paper
    except Exception as e:
        logger.error(f"Failed to fetch arXiv paper {arxiv_id}: {e}")

    return None

