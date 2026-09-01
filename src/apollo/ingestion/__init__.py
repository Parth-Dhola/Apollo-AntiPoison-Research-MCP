"""Ingestion clients for multi-source academic, code, and web data."""

from apollo.ingestion.arxiv_client import search_arxiv, fetch_arxiv_paper
from apollo.ingestion.semantic_scholar import search_semantic_scholar
from apollo.ingestion.github_client import search_github_repos
from apollo.ingestion.web_search import search_duckduckgo

__all__ = [
    "search_arxiv",
    "fetch_arxiv_paper",
    "search_semantic_scholar",
    "search_github_repos",
    "search_duckduckgo"
]

