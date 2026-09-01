"""schemas.py — Pydantic models with standard dataclass fallback for Apollo MCP Server."""

from enum import Enum
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    from dataclasses import dataclass, field

    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "value"):
                    res[k] = v.value
                elif isinstance(v, list):
                    res[k] = [item.model_dump() if hasattr(item, "model_dump") else item for item in v]
                elif hasattr(v, "model_dump"):
                    res[k] = v.model_dump()
                else:
                    res[k] = v
            return res

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return field(default_factory=default_factory)
        return field(default=default)


class QueryIntent(str, Enum):
    ACADEMIC_PAPER = "ACADEMIC_PAPER"
    CODE_IMPLEMENTATION = "CODE_IMPLEMENTATION"
    DEEP_THEORY = "DEEP_THEORY"
    GENERAL_WEB = "GENERAL_WEB"
    HYBRID = "HYBRID"


class ToolSelectionResult(BaseModel):
    query: str = ""
    intent: QueryIntent = QueryIntent.GENERAL_WEB
    recommended_tools: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    keywords: List[str] = Field(default_factory=list)
    reasoning: str = ""


class PaperMetadata(BaseModel):
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    citation_count: Optional[int] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    source: str = "arxiv"


class PaperSectionContext(BaseModel):
    arxiv_id: str = ""
    title: str = ""
    section_name: str = ""
    content: str = ""
    latex_equations: List[str] = Field(default_factory=list)
    token_count: int = 0


class CodeSnippetResult(BaseModel):
    repo_name: str = ""
    repo_url: str = ""
    file_path: str = ""
    language: Optional[str] = None
    stars: int = 0
    snippet: str = ""
    description: Optional[str] = None


class GroundedContextSnippet(BaseModel):
    source: str = "web"
    title: str = ""
    url: Optional[str] = None
    content: str = ""
    relevance_score: float = 0.0
    citation_meta: Dict[str, Any] = Field(default_factory=dict)
    sanitized: bool = True
    trust_level: str = "verified_clean"


class ResearchContextResult(BaseModel):
    query: str = ""
    intent: QueryIntent = QueryIntent.GENERAL_WEB
    snippets: List[GroundedContextSnippet] = Field(default_factory=list)
    total_sources_consulted: int = 0
    formatted_context: str = ""
    anti_poison_status: str = "passed"
    execution_time_ms: float = 0.0

