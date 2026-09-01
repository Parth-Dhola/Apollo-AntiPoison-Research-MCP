"""tool_selector.py — Standalone script & engine for zero-cost Bag-of-Words query intent routing."""

import argparse
import json
import re
import sys
from typing import Dict, List, Set

from apollo.models.schemas import QueryIntent, ToolSelectionResult

# ── Domain Vocabulary Bag-of-Words ──────────────────────────────────────────
_ACADEMIC_VOCAB: Set[str] = {
    "paper", "papers", "arxiv", "doi", "author", "authors", "citation", "citations",
    "theorem", "proof", "lemma", "proposition", "corollary", "survey", "literature",
    "sota", "state-of-the-art", "ablation", "dataset", "benchmark", "loss", "gradient",
    "architecture", "attention", "transformer", "diffusion", "latent", "embedding",
    "formulation", "convergence", "generalization", "empirical", "hypothesis", "evaluate",
    "accuracy", "f1", "bleu", "rouge", "perplexity", "conference", "journal", "neurips",
    "iclr", "icml", "cvpr", "emnlp", "acl", "aaai", "sigir", "kdd", "mechanism"
}

_EXPLICIT_CODE_VOCAB: Set[str] = {
    "github", "repo", "repository", "implementation", "implement", "snippet",
    "script", "pytorch", "torch", "tensorflow", "keras", "jax", "cuda", "c++",
    "rust", "golang", "npm", "pip", "docker", "dockerfile", "git", "commit",
    "pull", "pr", "clone", "fork", "bug", "issue", "debug", "optimizer",
    "dataloader", "checkpoint", "onnx", "quantization", "gguf", "llama.cpp"
}

_THEORY_VOCAB: Set[str] = {
    "equation", "equations", "formula", "derivation", "derive", "complexity", "big-o",
    "asymptotic", "runtime", "bound", "upper-bound", "lower-bound", "stochastic",
    "expectation", "variance", "distribution", "matrix", "eigenvalue", "eigenvector",
    "jacobian", "hessian", "convex", "optimization", "lagrangian", "dual", "primal"
}

_WEB_VOCAB: Set[str] = {
    "news", "latest", "recent", "release", "released", "announced", "launch", "today",
    "yesterday", "company", "event", "pricing", "cost", "schedule", "ceo", "founder",
    "market", "stock", "documentation", "guide", "tutorial", "blog", "post", "site"
}

_ARXIV_ID_PATTERN = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")


class ToolSelector:
    """
    Zero-cost Bag-of-Words (BoW) & heuristic query intent classifier and tool recommender.
    Requires $0 in external API costs, running in < 1ms on any CPU.
    """

    def __init__(self):
        self.academic_vocab = _ACADEMIC_VOCAB
        self.code_vocab = _EXPLICIT_CODE_VOCAB
        self.theory_vocab = _THEORY_VOCAB
        self.web_vocab = _WEB_VOCAB

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\.-]+\b", text.lower())

    def analyze_intent(self, query: str) -> ToolSelectionResult:
        if not query or not query.strip():
            return ToolSelectionResult(
                query=query,
                intent=QueryIntent.GENERAL_WEB,
                recommended_tools=["fallback_web_search"],
                confidence=0.5,
                keywords=[],
                reasoning="Empty query defaulted to general web search."
            )

        # 1. Check for exact arXiv ID or DOI pattern
        arxiv_match = _ARXIV_ID_PATTERN.findall(query)
        if arxiv_match:
            return ToolSelectionResult(
                query=query,
                intent=QueryIntent.ACADEMIC_PAPER,
                recommended_tools=["fetch_paper_deep_context", "search_academic_papers"],
                confidence=0.98,
                keywords=arxiv_match,
                reasoning=f"Explicit arXiv ID detected: {arxiv_match[0]}"
            )

        tokens = self.tokenize(query)
        token_set = set(tokens)

        # 2. Compute BoW overlap frequencies
        academic_matches = token_set.intersection(self.academic_vocab)
        code_matches = token_set.intersection(self.code_vocab)
        theory_matches = token_set.intersection(self.theory_vocab)
        web_matches = token_set.intersection(self.web_vocab)

        score_academic = len(academic_matches) * 1.5 + len(theory_matches) * 1.2
        score_code = len(code_matches) * 1.5
        score_web = len(web_matches) * 1.0

        all_keywords = list(academic_matches | code_matches | theory_matches | web_matches)

        # 3. Determine Intent & Recommended Tools
        # If explicit code keywords and strong academic keywords both exist
        if len(code_matches) >= 1 and (len(academic_matches) >= 1 or len(theory_matches) >= 1):
            intent = QueryIntent.HYBRID
            tools = ["search_academic_papers", "search_repo_implementations"]
            confidence = min(0.95, 0.6 + 0.1 * (score_academic + score_code))
            reasoning = f"Hybrid query matching academic concepts ({', '.join(academic_matches)}) and code keywords ({', '.join(code_matches)})."

        elif score_code > 0 and score_code > score_academic:
            intent = QueryIntent.CODE_IMPLEMENTATION
            tools = ["search_repo_implementations", "fallback_web_search"]
            confidence = min(0.95, 0.6 + 0.15 * score_code)
            reasoning = f"Code implementation query matching keywords: {', '.join(code_matches)}"

        elif score_academic > 0:
            intent = QueryIntent.ACADEMIC_PAPER
            tools = ["search_academic_papers", "fetch_paper_deep_context"]
            confidence = min(0.95, 0.6 + 0.15 * score_academic)
            reasoning = f"Academic research query matching concepts: {', '.join(academic_matches | theory_matches)}"

        elif score_web > 0:
            intent = QueryIntent.GENERAL_WEB
            tools = ["fallback_web_search"]
            confidence = min(0.90, 0.5 + 0.15 * score_web)
            reasoning = f"General web lookup matching keywords: {', '.join(web_matches)}"

        else:
            # Default to academic + web search fallback if no specific keywords matched
            intent = QueryIntent.ACADEMIC_PAPER
            tools = ["search_academic_papers", "fallback_web_search"]
            confidence = 0.55
            reasoning = "Generic query defaulted to academic search with web fallback."

        return ToolSelectionResult(
            query=query,
            intent=intent,
            recommended_tools=tools,
            confidence=round(confidence, 2),
            keywords=all_keywords,
            reasoning=reasoning
        )


_global_selector: ToolSelector = ToolSelector()


def select_tools_for_query(query: str) -> ToolSelectionResult:
    """Convenience function for standalone tool selection."""
    return _global_selector.analyze_intent(query)


def cli_main():
    """Standalone CLI entry point for testing and executing tool routing."""
    parser = argparse.ArgumentParser(description="Apollo Standalone Tool Selector & Intent Router")
    parser.add_argument("query", nargs="*", help="Query string to analyze")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    query_str = " ".join(args.query).strip() if args.query else ""

    if not query_str:
        print("Usage: python -m apollo.router.tool_selector \"Your query here\"")
        print("Example: python -m apollo.router.tool_selector \"How to implement LoRA in PyTorch?\"")
        sys.exit(1)

    result = select_tools_for_query(query_str)

    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print("=" * 60)
        print(" Apollo Tool Selector & Intent Router")
        print("=" * 60)
        print(f"Query:             {result.query}")
        print(f"Predicted Intent:  {result.intent.value}")
        print(f"Confidence:        {result.confidence * 100:.0f}%")
        print(f"Keywords Matched:  {', '.join(result.keywords) if result.keywords else '(none)'}")
        print(f"Recommended Tools: {', '.join(result.recommended_tools)}")
        print(f"Reasoning:         {result.reasoning}")
        print("=" * 60)


if __name__ == "__main__":
    cli_main()

