"""tool_rag.py — RAG-based Semantic Tool Indexer & Context Budgeting Engine.

Indexes tool capability profiles and uses BM25 & Semantic Scoring to dynamically
select ONLY the necessary tools for a query, preventing context overload and redundant API calls.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from apollo.guardrail_rag.bow_bm25 import BM25Ranker


@dataclass
class ToolProfile:
    name: str
    description: str
    keywords: List[str]
    sample_queries: List[str]
    category: str
    token_cost: str  # "low", "medium", "high"
    min_score_threshold: float = 0.25

    def get_searchable_document(self) -> str:
        """Create rich semantic document for RAG indexing."""
        return (
            f"Tool: {self.name}\n"
            f"Category: {self.category}\n"
            f"Description: {self.description}\n"
            f"Keywords: {' '.join(self.keywords)}\n"
            f"Sample Queries: {' '.join(self.sample_queries)}"
        )


# ── Registered Tool Profiles ────────────────────────────────────────────────
_TOOL_PROFILES: List[ToolProfile] = [
    ToolProfile(
        name="search_academic_papers",
        description="Search peer-reviewed academic literature, arXiv preprints, citations, methodology, mathematical formulations, and theoretical theorems.",
        keywords=["paper", "papers", "arxiv", "doi", "theorem", "proof", "ablation", "sota", "survey", "literature", "citations", "loss", "convergence", "attention", "transformer", "diffusion", "latent", "dataset", "benchmark"],
        sample_queries=[
            "Survey of vision transformers",
            "Convergence proof for Adam optimizer",
            "Attention mechanism mathematical formulation",
            "Recent papers on diffusion state space models"
        ],
        category="academic",
        token_cost="medium",
        min_score_threshold=0.30
    ),
    ToolProfile(
        name="fetch_paper_deep_context",
        description="Fetch deep technical equations, abstract, and sections for a specific arXiv ID (e.g. 2205.14135, 2307.08691).",
        keywords=["arxiv_id", "2205.14135", "2307.08691", "paper id", "arxiv paper", "deep context", "fetch paper"],
        sample_queries=[
            "Explain arXiv paper 2205.14135",
            "What is in paper 2307.08691v2?",
            "Fetch equations for arxiv:2106.09685"
        ],
        category="academic_exact",
        token_cost="high",
        min_score_threshold=0.50
    ),
    ToolProfile(
        name="search_repo_implementations",
        description="Search verified GitHub open-source repositories for code snippets, PyTorch / CUDA implementations, models, and scripts.",
        keywords=["github", "repo", "repository", "code", "implementation", "implement", "pytorch", "torch", "cuda", "script", "snippet", "package", "class", "function", "clone", "fork", "stars", "onnx", "llama.cpp"],
        sample_queries=[
            "PyTorch implementation of LoRA linear layer",
            "CUDA kernel implementation of FlashAttention",
            "GitHub repository for YOLOv8",
            "Code snippet for multi-head self-attention in Python"
        ],
        category="code",
        token_cost="medium",
        min_score_threshold=0.30
    ),
    ToolProfile(
        name="fallback_web_search",
        description="Fallback search for recent tech news, product launches, pricing, documentation, tutorials, and general web information.",
        keywords=["news", "latest", "recent", "release", "released", "announced", "launch", "today", "pricing", "cost", "documentation", "guide", "tutorial", "blog", "founder", "ceo"],
        sample_queries=[
            "OpenAI GPT-4o release date and pricing",
            "Latest news on Nvidia Blackwell GPUs",
            "How to install PyTorch on Mac M3",
            "LangChain official documentation tutorial"
        ],
        category="web",
        token_cost="low",
        min_score_threshold=0.20
    )
]


class ToolRAGSelector:
    """
    RAG-based Semantic Tool Indexer.
    Indexes tool profiles and computes relevance scores to dynamically prune unnecessary tools.
    """

    def __init__(self, profiles: Optional[List[ToolProfile]] = None):
        self.profiles = profiles or _TOOL_PROFILES
        self.doc_texts = [p.get_searchable_document() for p in self.profiles]
        self.bm25 = BM25Ranker(k1=1.5, b=0.75)
        self.bm25.fit(self.doc_texts)

    def select_tools(
        self,
        query: str,
        max_tools: int = 2,
        strict_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform RAG matching over tool capability documents.
        Returns a ranked list of tool selections with scores, pruning tools that don't meet relevance thresholds.
        """
        if not query or not query.strip():
            return [{
                "tool_name": "fallback_web_search",
                "score": 1.0,
                "reason": "Default web fallback for empty query."
            }]

        # Check for explicit arXiv ID match -> short-circuit to ONLY deep context fetch
        import re
        if re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", query):
            return [
                {
                    "tool_name": "fetch_paper_deep_context",
                    "score": 1.0,
                    "reason": "Exact arXiv ID detected. Short-circuiting all other tools to prevent context overload."
                }
            ]

        scores = self.bm25.score(query)
        selected = []

        # Find max score to normalize
        max_score = scores[0][1] if scores and scores[0][1] > 0 else 1.0

        for doc_idx, raw_score in scores:
            profile = self.profiles[doc_idx]
            norm_score = raw_score / max_score if max_score > 0 else 0.0

            # Prune if score doesn't pass tool's minimum threshold
            if strict_threshold and norm_score < profile.min_score_threshold:
                continue

            selected.append({
                "tool_name": profile.name,
                "score": round(norm_score, 3),
                "category": profile.category,
                "token_cost": profile.token_cost,
                "reason": f"Matched tool capabilities with relevance score {norm_score:.2f}."
            })

            if len(selected) >= max_tools:
                break

        # Fallback to web search if all tools were pruned
        if not selected:
            selected.append({
                "tool_name": "fallback_web_search",
                "score": 0.5,
                "category": "web",
                "token_cost": "low",
                "reason": "No high-confidence specialized tools matched; falling back to web search."
            })

        return selected


_global_tool_rag = ToolRAGSelector()


def rank_tools_for_query(query: str, max_tools: int = 2) -> List[Dict[str, Any]]:
    """Convenience function for RAG-based tool selection."""
    return _global_tool_rag.select_tools(query, max_tools=max_tools)

