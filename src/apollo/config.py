"""config.py — Global settings for Apollo MCP Server with lightweight fallback."""

import os
from typing import Literal, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )

        # General
        APP_NAME: str = "apollo-mcp"
        DEBUG: bool = False

        # Server Transport
        APOLLO_TRANSPORT: Literal["stdio", "sse"] = "stdio"
        APOLLO_HOST: str = "0.0.0.0"
        APOLLO_PORT: int = 8080

        # Free API Rate Limits & Credentials
        GITHUB_TOKEN: Optional[str] = None
        SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
        USER_AGENT: str = "Apollo-Research-MCP/0.1.0 (academic-context-tool)"

        # Caching
        CACHE_DIR: str = ".apollo_cache"
        CACHE_TTL_SECONDS: int = 86400  # 24 hours

        # Guardrail RAG & Reranker Settings
        MAX_CANDIDATE_SNIPPETS: int = 15
        TOP_K_GROUNDED_SNIPPETS: int = 3
        RERANKER_MODEL: str = "ms-marco-TinyBERT-L-2-v2"
        BM25_K1: float = 1.5
        BM25_B: float = 0.75

        # Anti-Poisoning & Sanitization
        ENABLE_PROMPT_INJECTION_FILTER: bool = True
        ENABLE_LATEX_NORMALIZATION: bool = True
        ENABLE_NOISE_REDUCTION: bool = True
        MAX_SNIPPET_LENGTH: int = 1200

except ImportError:
    class Settings:
        APP_NAME: str = "apollo-mcp"
        DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
        APOLLO_TRANSPORT: str = os.getenv("APOLLO_TRANSPORT", "stdio")
        APOLLO_HOST: str = os.getenv("APOLLO_HOST", "0.0.0.0")
        APOLLO_PORT: int = int(os.getenv("APOLLO_PORT", "8080"))
        GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
        SEMANTIC_SCHOLAR_API_KEY: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        USER_AGENT: str = os.getenv("USER_AGENT", "Apollo-Research-MCP/0.1.0 (academic-context-tool)")
        CACHE_DIR: str = os.getenv("CACHE_DIR", ".apollo_cache")
        CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
        MAX_CANDIDATE_SNIPPETS: int = int(os.getenv("MAX_CANDIDATE_SNIPPETS", "15"))
        TOP_K_GROUNDED_SNIPPETS: int = int(os.getenv("TOP_K_GROUNDED_SNIPPETS", "3"))
        RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
        BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
        BM25_B: float = float(os.getenv("BM25_B", "0.75"))
        ENABLE_PROMPT_INJECTION_FILTER: bool = os.getenv("ENABLE_PROMPT_INJECTION_FILTER", "true").lower() == "true"
        ENABLE_LATEX_NORMALIZATION: bool = os.getenv("ENABLE_LATEX_NORMALIZATION", "true").lower() == "true"
        ENABLE_NOISE_REDUCTION: bool = os.getenv("ENABLE_NOISE_REDUCTION", "true").lower() == "true"
        MAX_SNIPPET_LENGTH: int = int(os.getenv("MAX_SNIPPET_LENGTH", "1200"))


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

