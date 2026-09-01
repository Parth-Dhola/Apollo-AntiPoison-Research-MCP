"""reranker.py — Low-latency CPU cross-encoder and BM25 hybrid reranker with Source Authority Weighting."""

import logging
from typing import List, Dict, Any, Union
from apollo.guardrail_rag.bow_bm25 import BM25Ranker
from apollo.models.schemas import GroundedContextSnippet

logger = logging.getLogger("apollo.reranker")

_ranker_instance = None

# ── Source Credibility / Authority Hierarchy ─────────────────────────────────
# Academic peer-reviewed papers are Tier 1 (1.0).
# Verified code implementations are Tier 2 (0.90).
# Wikipedia is crowd-sourced encyclopedia Tier 3 (0.65 - Deprioritized for deep research).
# General web is Tier 4 (0.55).
SOURCE_AUTHORITY_WEIGHTS: Dict[str, float] = {
    "arxiv": 1.0,
    "semantic_scholar": 1.0,
    "github": 0.90,
    "wikipedia": 0.65,
    "web": 0.55
}


def _get_flashrank():
    global _ranker_instance
    if _ranker_instance is None:
        try:
            from flashrank import Ranker
            _ranker_instance = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=".flashrank_cache")
            logger.debug("FlashRank CPU Ranker loaded successfully.")
        except Exception as e:
            logger.info(f"FlashRank not initialized ({e}), using BM25 ranking.")
            _ranker_instance = False
    return _ranker_instance if _ranker_instance is not False else None


def rank_snippets(
    query: str,
    snippets: List[Union[GroundedContextSnippet, Dict[str, Any]]],
    top_k: int = 3,
    apply_source_authority: bool = True
) -> List[GroundedContextSnippet]:
    """
    Rerank candidate snippets against the query using FlashRank or BM25,
    applying Source Authority Hierarchy multipliers to prioritize peer-reviewed literature.
    """
    if not snippets or not query:
        return []

    # Standardize input list to GroundedContextSnippet
    converted: List[GroundedContextSnippet] = []
    for s in snippets:
        if isinstance(s, GroundedContextSnippet):
            converted.append(s)
        elif isinstance(s, dict):
            converted.append(GroundedContextSnippet(**s))

    if len(converted) == 1:
        weight = SOURCE_AUTHORITY_WEIGHTS.get(converted[0].source, 0.6) if apply_source_authority else 1.0
        converted[0].relevance_score = 1.0 * weight
        return converted

    scored_snippets: List[GroundedContextSnippet] = []

    # 1. Try FlashRank Cross-Encoder
    flashrank = _get_flashrank()
    if flashrank is not None:
        try:
            from flashrank import RerankRequest
            passages = [
                {"id": idx, "text": f"{s.title}\n{s.content}"}
                for idx, s in enumerate(converted)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = flashrank.rerank(rerank_request)

            for r in results:
                idx = int(r["id"])
                snippet = converted[idx]
                raw_score = float(r.get("score", 0.0))
                auth_multiplier = SOURCE_AUTHORITY_WEIGHTS.get(snippet.source, 0.6) if apply_source_authority else 1.0
                snippet.relevance_score = round(raw_score * auth_multiplier, 4)
                scored_snippets.append(snippet)

            # Sort by authority-weighted relevance score
            scored_snippets.sort(key=lambda x: x.relevance_score, reverse=True)
            return scored_snippets[:top_k]
        except Exception as e:
            logger.warning(f"FlashRank reranking failed ({e}), falling back to BM25.")

    # 2. Fallback: BM25 Ranker with Source Authority Weighting
    texts = [f"{s.title}\n{s.content}" for s in converted]
    bm25 = BM25Ranker()
    bm25.fit(texts)
    scores = bm25.score(query)

    max_bm25 = scores[0][1] if scores and scores[0][1] > 0 else 1.0

    for idx, raw_score in scores:
        snippet = converted[idx]
        norm_score = raw_score / max_bm25 if max_bm25 > 0 else 0.0
        auth_multiplier = SOURCE_AUTHORITY_WEIGHTS.get(snippet.source, 0.6) if apply_source_authority else 1.0
        snippet.relevance_score = round(norm_score * auth_multiplier, 4)
        scored_snippets.append(snippet)

    scored_snippets.sort(key=lambda x: x.relevance_score, reverse=True)
    return scored_snippets[:top_k]
